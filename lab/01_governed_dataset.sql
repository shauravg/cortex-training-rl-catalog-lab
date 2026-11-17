-- =====================================================================
-- Step 1: Governed training data for marketplace listing review
-- Synthetic listings with planted PII -> classify -> redact -> mask -> split
-- Replace <DB>, <SCHEMA>, <WH>, <ANALYST_ROLE> before running.
-- =====================================================================
USE WAREHOUSE <WH>;
USE SCHEMA <DB>.<SCHEMA>;

-- ---------------------------------------------------------------------
-- 1a. Synthetic raw listings (demo data). ~6,000 rows, fixed seed.
--     Gold labels: category, brand, violation.
--     Violations: none | off_platform_contact | prohibited_item | counterfeit_signal
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE RAW_LISTINGS AS
WITH base AS (
  SELECT
    SEQ4()                                             AS listing_id,
    UNIFORM(0, 4, RANDOM(11))                          AS c_idx,
    UNIFORM(0, 3, RANDOM(12))                          AS b_idx,
    UNIFORM(0, 99, RANDOM(13))                         AS v_roll,
    UNIFORM(0, 7, RANDOM(14))                          AS n_idx,
    UNIFORM(1000, 9999, RANDOM(15))                    AS phone_tail,
    UNIFORM(5, 500, RANDOM(16))                        AS price
  FROM TABLE(GENERATOR(ROWCOUNT => 6000))
),
dict AS (
  SELECT
    ARRAY_CONSTRUCT('electronics','apparel','home_kitchen','toys','beauty')            AS cats,
    ARRAY_CONSTRUCT(
      ARRAY_CONSTRUCT('Sony','Anker','Logitech','JBL'),
      ARRAY_CONSTRUCT('Levi''s','Patagonia','Uniqlo','Carhartt'),
      ARRAY_CONSTRUCT('OXO','Lodge','Pyrex','KitchenAid'),
      ARRAY_CONSTRUCT('LEGO','Hasbro','Mattel','Melissa & Doug'),
      ARRAY_CONSTRUCT('CeraVe','Olaplex','Neutrogena','The Ordinary'))                AS brands,
    ARRAY_CONSTRUCT('wireless earbuds','hiking jacket','cast iron skillet',
                    'building set','moisturizing cream')                               AS nouns,
    ARRAY_CONSTRUCT('Maria Lopez','James Chen','Aisha Khan','Tom Becker',
                    'Priya Nair','Diego Santos','Emma Wright','Kenji Sato')            AS sellers
)
SELECT
  b.listing_id,
  dict.sellers[b.n_idx]::STRING                                         AS seller_name,
  dict.cats[b.c_idx]::STRING                                            AS gold_category,
  dict.brands[b.c_idx][b.b_idx]::STRING                                 AS gold_brand,
  CASE WHEN b.v_roll < 70 THEN 'none'
       WHEN b.v_roll < 82 THEN 'off_platform_contact'
       WHEN b.v_roll < 91 THEN 'counterfeit_signal'
       ELSE 'prohibited_item' END                                       AS gold_violation,
  b.price,
  -- Title + description with the violation planted in the text
  dict.brands[b.c_idx][b.b_idx]::STRING || ' ' || dict.nouns[b.c_idx]::STRING
    || ' - like new'                                                    AS title,
  'Selling my ' || dict.brands[b.c_idx][b.b_idx]::STRING || ' '
    || dict.nouns[b.c_idx]::STRING || ', used twice, ships in 2 days. Listed by '
    || dict.sellers[b.n_idx]::STRING || '. '
    || CASE
         WHEN b.v_roll < 70 THEN 'Price is firm. Message me here with questions.'
         WHEN b.v_roll < 82 THEN 'Skip the fees, text me at (415) 555-' || b.phone_tail
                              || ' or email ' || LOWER(SPLIT_PART(dict.sellers[b.n_idx]::STRING,' ',1))
                              || b.phone_tail || '@gmail.com for a better price.'
         WHEN b.v_roll < 91 THEN 'Factory replica, 1:1 quality, comes without original box. Only $'
                              || ROUND(b.price * 0.2) || '.'
         ELSE 'Also includes a spare lithium battery pack and pepper spray keychain, no ID needed.'
       END                                                              AS description
FROM base b, dict;

-- ---------------------------------------------------------------------
-- 1b. Detect and tag PII (column-level governance)
-- ---------------------------------------------------------------------
CALL SYSTEM$CLASSIFY('<DB>.<SCHEMA>.RAW_LISTINGS', {'auto_tag': true});

-- Free text hides PII the column classifier cannot see. Check with AI_REDACT detect mode.
SELECT listing_id,
       AI_REDACT(input => description, mode => 'detect') AS pii_spans
FROM RAW_LISTINGS
WHERE gold_violation = 'off_platform_contact'
LIMIT 5;

-- ---------------------------------------------------------------------
-- 1c. Mask raw PII for everyone except the data owner role
-- ---------------------------------------------------------------------
-- Use CURRENT_ROLE(), not IS_ROLE_IN_SESSION(): with secondary roles set to ALL
-- (a common default), IS_ROLE_IN_SESSION is true for anyone who merely HOLDS the
-- owner role, so every one of their sessions would see raw PII.
CREATE OR REPLACE MASKING POLICY PII_TEXT_MASK AS (val STRING) RETURNS STRING ->
  CASE WHEN CURRENT_ROLE() = '<ANALYST_ROLE>' THEN val
       ELSE '***MASKED***' END;

ALTER TABLE RAW_LISTINGS MODIFY COLUMN seller_name SET MASKING POLICY PII_TEXT_MASK;
ALTER TABLE RAW_LISTINGS MODIFY COLUMN description SET MASKING POLICY PII_TEXT_MASK;

-- ---------------------------------------------------------------------
-- 1d. Redacted training view. PII becomes typed placeholders, so the
--     model still learns "contact info in a listing => violation"
--     without ever seeing a real email or phone number.
--     Run as <ANALYST_ROLE> so the source text is unmasked for redaction.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE LISTINGS_REDACTED AS
SELECT
  listing_id,
  title,
  AI_REDACT(description, ['NAME','EMAIL','PHONE_NUMBER','ADDRESS']) AS description_redacted,
  price,
  gold_category, gold_brand, gold_violation,
  IFF(ABS(HASH(listing_id)) % 10 < 8, 'train', 'test')                AS split
FROM RAW_LISTINGS;

-- Governance check: no raw email or phone survives redaction. Expect 0.
SELECT COUNT(*) AS leaked_rows
FROM LISTINGS_REDACTED
WHERE REGEXP_LIKE(description_redacted, '.*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[a-z]{2,}.*')
   OR REGEXP_LIKE(description_redacted, '.*\\(\\d{3}\\) \\d{3}-\\d{4}.*');

-- Label balance per split
SELECT split, gold_violation, COUNT(*) FROM LISTINGS_REDACTED GROUP BY 1, 2 ORDER BY 1, 2;

-- ---------------------------------------------------------------------
-- 1e. Export train/test as JSONL to a stage for the RL driver
-- ---------------------------------------------------------------------
CREATE STAGE IF NOT EXISTS RL_DATA;

COPY INTO @RL_DATA/listings_train.jsonl
FROM (
  SELECT OBJECT_CONSTRUCT(
    'listing_id', listing_id,
    'title', title,
    'description', description_redacted,
    'price', price,
    'label', OBJECT_CONSTRUCT('category', gold_category, 'brand', gold_brand,
                              'violation', gold_violation))
  FROM LISTINGS_REDACTED WHERE split = 'train')
FILE_FORMAT = (TYPE = JSON COMPRESSION = NONE) SINGLE = TRUE OVERWRITE = TRUE;

COPY INTO @RL_DATA/listings_test.jsonl
FROM (
  SELECT OBJECT_CONSTRUCT(
    'listing_id', listing_id,
    'title', title,
    'description', description_redacted,
    'price', price,
    'label', OBJECT_CONSTRUCT('category', gold_category, 'brand', gold_brand,
                              'violation', gold_violation))
  FROM LISTINGS_REDACTED WHERE split = 'test')
FILE_FORMAT = (TYPE = JSON COMPRESSION = NONE) SINGLE = TRUE OVERWRITE = TRUE;

-- Pull locally:  snow stage copy @<DB>.<SCHEMA>.RL_DATA ./data/
