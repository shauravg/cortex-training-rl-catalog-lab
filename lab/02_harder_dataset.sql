-- =====================================================================
-- Step 1b (v2): a dataset that can actually test generalization
--
-- v1 (01_governed_dataset.sql) reuses ONE phrasing per violation and splits
-- rows at random, so the test set shares every template with training.
-- v2 fixes that:
--   * 8 phrasings per class; phrasings 6-7 appear ONLY in the test split
--   * tricky clean listings (licensed "replica" props, AA batteries,
--     "phone" mounts) that share words with violations
--   * every description rewritten by AI_COMPLETE so no two read alike
--   * PII is checked again AFTER paraphrasing, then redacted
-- Replace <DB>, <SCHEMA>, <WH>, <ANALYST_ROLE> before running.
-- =====================================================================
USE WAREHOUSE <WH>;
USE SCHEMA <DB>.<SCHEMA>;

-- ---------------------------------------------------------------------
-- 2a. Phrasing bank: class x template_id. {PHONE} {EMAIL} {LOW} are filled per row.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE PHRASES (violation STRING, template_id INT, phrase STRING) AS
SELECT * FROM VALUES
  ('none',0,'Price is firm. Message me here with questions.'),
  ('none',1,'Open to reasonable offers through the app. Bundle discounts available.'),
  ('none',2,'Includes two AA batteries and the original box.'),
  ('none',3,'Officially licensed replica of the movie prop, certificate of authenticity included.'),
  ('none',4,'Works with iPhone and Android phones, no adapter needed.'),
  ('none',5,'Use the Ask the Seller button, I usually reply within an hour.'),
  ('none',6,'Authorized reseller, receipt available on request through this site.'),
  ('none',7,'Comes with a phone mount, and the mic call quality is excellent.'),
  ('off_platform_contact',0,'Skip the fees, text me at {PHONE} or email {EMAIL} for a better price.'),
  ('off_platform_contact',1,'Messages here are slow, reach me on {PHONE} instead.'),
  ('off_platform_contact',2,'Email {EMAIL} and I will send a PayPal invoice, cheaper than checkout.'),
  ('off_platform_contact',3,'Call {PHONE} after 6pm to arrange cash pickup.'),
  ('off_platform_contact',4,'For 15% off contact me directly at {EMAIL}.'),
  ('off_platform_contact',5,'WhatsApp me at {PHONE}, I do not check this app.'),
  ('off_platform_contact',6,'Let us deal outside the app and save us both money, {EMAIL}.'),
  ('off_platform_contact',7,'Ping my cell {PHONE} and we can do Venmo.'),
  ('counterfeit_signal',0,'Factory replica, 1:1 quality, no original box. Only ${LOW}.'),
  ('counterfeit_signal',1,'Mirror quality copy, looks identical to retail, ${LOW} firm.'),
  ('counterfeit_signal',2,'Unbranded factory overrun with the logo added, way cheaper than the store.'),
  ('counterfeit_signal',3,'AAA grade, nobody can tell the difference. No receipt.'),
  ('counterfeit_signal',4,'Inspired version, same materials as the real one, generic tags.'),
  ('counterfeit_signal',5,'Bought in bulk from a wholesaler, not authentic but great quality.'),
  ('counterfeit_signal',6,'Same factory as the brand, just without the brand markup, ${LOW}.'),
  ('counterfeit_signal',7,'High-end dupe, honestly better than the original.'),
  ('prohibited_item',0,'Also includes a spare lithium battery pack and a pepper spray keychain, no ID needed.'),
  ('prohibited_item',1,'Throwing in some prescription sleeping pills I no longer need.'),
  ('prohibited_item',2,'Comes with a hunting knife and a box of ammo, local pickup only.'),
  ('prohibited_item',3,'Includes a vape cartridge set, no age check.'),
  ('prohibited_item',4,'Bundled with a stun gun, message for details.'),
  ('prohibited_item',5,'Also selling my old prescription contact lenses, unopened.'),
  ('prohibited_item',6,'Includes leftover fireworks from July, unopened.'),
  ('prohibited_item',7,'Comes with a used airbag module pulled from a car.');

-- ---------------------------------------------------------------------
-- 2b. Raw v2 listings. 5,000 rows, fixed seed. Split by TEMPLATE, not by row.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE RAW_LISTINGS_V2 AS
WITH base AS (
  SELECT SEQ4() AS listing_id,
         UNIFORM(0, 4, RANDOM(21)) AS c_idx, UNIFORM(0, 3, RANDOM(22)) AS b_idx,
         UNIFORM(0, 2, RANDOM(23)) AS n_idx, UNIFORM(0, 99, RANDOM(24)) AS v_roll,
         UNIFORM(0, 7, RANDOM(25)) AS t_idx, UNIFORM(0, 7, RANDOM(26)) AS s_idx,
         UNIFORM(1000, 9999, RANDOM(27)) AS phone_tail, UNIFORM(5, 500, RANDOM(28)) AS price
  FROM TABLE(GENERATOR(ROWCOUNT => 5000))
),
dict AS (
  SELECT
    ARRAY_CONSTRUCT('electronics','apparel','home_kitchen','toys','beauty') AS cats,
    ARRAY_CONSTRUCT(
      ARRAY_CONSTRUCT('Sony','Anker','Logitech','JBL'),
      ARRAY_CONSTRUCT('Levi''s','Patagonia','Uniqlo','Carhartt'),
      ARRAY_CONSTRUCT('OXO','Lodge','Pyrex','KitchenAid'),
      ARRAY_CONSTRUCT('LEGO','Hasbro','Mattel','Melissa & Doug'),
      ARRAY_CONSTRUCT('CeraVe','Olaplex','Neutrogena','The Ordinary')) AS brands,
    ARRAY_CONSTRUCT(
      ARRAY_CONSTRUCT('wireless earbuds','bluetooth speaker','webcam'),
      ARRAY_CONSTRUCT('denim jacket','rain shell','work pants'),
      ARRAY_CONSTRUCT('cast iron skillet','mixing bowl set','stand mixer'),
      ARRAY_CONSTRUCT('building set','board game','doll house'),
      ARRAY_CONSTRUCT('moisturizing cream','hair serum','sunscreen')) AS nouns,
    ARRAY_CONSTRUCT('Maria Lopez','James Chen','Aisha Khan','Tom Becker',
                    'Priya Nair','Diego Santos','Emma Wright','Kenji Sato') AS sellers
)
SELECT
  b.listing_id,
  dict.sellers[b.s_idx]::STRING AS seller_name,
  dict.cats[b.c_idx]::STRING AS gold_category,
  dict.brands[b.c_idx][b.b_idx]::STRING AS gold_brand,
  CASE WHEN b.v_roll < 65 THEN 'none' WHEN b.v_roll < 77 THEN 'off_platform_contact'
       WHEN b.v_roll < 89 THEN 'counterfeit_signal' ELSE 'prohibited_item' END AS gold_violation,
  b.t_idx AS template_id,
  IFF(b.t_idx >= 6, 'test', 'train') AS split,
  b.price,
  dict.brands[b.c_idx][b.b_idx]::STRING || ' ' || dict.nouns[b.c_idx][b.n_idx]::STRING AS title,
  '(415) 555-' || b.phone_tail AS phone,
  LOWER(SPLIT_PART(dict.sellers[b.s_idx]::STRING, ' ', 1)) || b.phone_tail || '@gmail.com' AS email,
  dict.brands[b.c_idx][b.b_idx]::STRING || ' ' || dict.nouns[b.c_idx][b.n_idx]::STRING AS item
FROM base b, dict;

-- Draft description = item + seller + phrase with placeholders filled
CREATE OR REPLACE TABLE RAW_LISTINGS_V2_DRAFT AS
SELECT r.*,
  'Selling my ' || r.item || ', listed by ' || r.seller_name || '. ' ||
  REPLACE(REPLACE(REPLACE(p.phrase, '{PHONE}', r.phone), '{EMAIL}', r.email),
          '{LOW}', ROUND(r.price * 0.2)::STRING) AS draft
FROM RAW_LISTINGS_V2 r JOIN PHRASES p
  ON p.violation = r.gold_violation AND p.template_id = r.template_id;

-- ---------------------------------------------------------------------
-- 2c. Paraphrase with AI_COMPLETE so wording varies row to row.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE RAW_LISTINGS_V2_PARA AS
SELECT d.*,
  TRIM(AI_COMPLETE('llama3.3-70b',
    'Rewrite this marketplace listing description in a casual, natural seller voice. '
    || 'Keep every fact, product, price, phone number, email address and person name exactly as written. '
    || 'Do not add new facts, warnings or contact details. One to three sentences. '
    || 'Return only the rewritten description.' || CHR(10) || CHR(10) || d.draft,
    {'temperature': 0.8, 'max_tokens': 160})) AS description
FROM RAW_LISTINGS_V2_DRAFT d;

-- Paraphrase QA: contact rows must still carry their phone or email,
-- other rows must not have gained any. Drop rows that fail.
SELECT gold_violation,
       COUNT(*) AS n,
       COUNT_IF(gold_violation = 'off_platform_contact'
                AND NOT (CONTAINS(description, phone) OR CONTAINS(description, email))) AS lost_contact,
       COUNT_IF(gold_violation <> 'off_platform_contact'
                AND REGEXP_LIKE(description, '.*(@|\\d{3}-\\d{4}).*')) AS gained_contact
FROM RAW_LISTINGS_V2_PARA GROUP BY 1 ORDER BY 1;

CREATE OR REPLACE TABLE RAW_LISTINGS_V2_FINAL AS
SELECT listing_id, seller_name, gold_category, gold_brand, gold_violation,
       template_id, split, price, title, description
FROM RAW_LISTINGS_V2_PARA
WHERE IFF(gold_violation = 'off_platform_contact',
          CONTAINS(description, phone) OR CONTAINS(description, email),
          NOT REGEXP_LIKE(description, '.*(@|\\d{3}-\\d{4}).*'));

-- ---------------------------------------------------------------------
-- 2d. Same governance as v1: tag, mask, redact with typed placeholders
-- ---------------------------------------------------------------------
CALL SYSTEM$CLASSIFY('<DB>.<SCHEMA>.RAW_LISTINGS_V2_FINAL', {'auto_tag': true});
ALTER TABLE RAW_LISTINGS_V2_FINAL MODIFY COLUMN seller_name SET MASKING POLICY PII_TEXT_MASK;
ALTER TABLE RAW_LISTINGS_V2_FINAL MODIFY COLUMN description SET MASKING POLICY PII_TEXT_MASK;

-- Prove it: a role that can read the table but is not the owner sees ***MASKED***.
-- (Run as a user whose session is NOT also carrying <ANALYST_ROLE>.)
CREATE ROLE IF NOT EXISTS CATALOG_RL_VIEWER;
GRANT USAGE ON DATABASE <DB> TO ROLE CATALOG_RL_VIEWER;
GRANT USAGE ON SCHEMA <DB>.<SCHEMA> TO ROLE CATALOG_RL_VIEWER;
GRANT SELECT ON TABLE RAW_LISTINGS_V2_FINAL TO ROLE CATALOG_RL_VIEWER;
-- USE ROLE CATALOG_RL_VIEWER;
-- SELECT seller_name, description FROM RAW_LISTINGS_V2_FINAL LIMIT 2;   -- ***MASKED***
-- USE ROLE <ANALYST_ROLE>;

CREATE OR REPLACE TABLE LISTINGS_V2_REDACTED AS
SELECT listing_id, title,
  AI_REDACT(description, ['NAME','EMAIL','PHONE_NUMBER','ADDRESS']) AS description_redacted,
  price, gold_category, gold_brand, gold_violation, template_id, split
FROM RAW_LISTINGS_V2_FINAL;

-- Governance check: expect all zeros
SELECT
  COUNT_IF(REGEXP_LIKE(description_redacted, '.*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[a-z]{2,}.*')) AS leaked_email,
  COUNT_IF(REGEXP_LIKE(description_redacted, '.*\\d{3}\\)? ?-?\\d{3}-\\d{4}.*')) AS leaked_phone,
  COUNT_IF(REGEXP_LIKE(description_redacted,
    '.*(Maria Lopez|James Chen|Aisha Khan|Tom Becker|Priya Nair|Diego Santos|Emma Wright|Kenji Sato).*')) AS leaked_name,
  COUNT(*) AS total
FROM LISTINGS_V2_REDACTED;

SELECT split, gold_violation, COUNT(*) FROM LISTINGS_V2_REDACTED GROUP BY 1, 2 ORDER BY 1, 2;

-- ---------------------------------------------------------------------
-- 2e. Export
-- ---------------------------------------------------------------------
COPY INTO @RL_DATA/listings_v2_train.jsonl
FROM (SELECT OBJECT_CONSTRUCT('listing_id', listing_id, 'title', title, 'description', description_redacted,
        'price', price, 'template_id', template_id,
        'label', OBJECT_CONSTRUCT('category', gold_category, 'brand', gold_brand, 'violation', gold_violation))
      FROM LISTINGS_V2_REDACTED WHERE split = 'train')
FILE_FORMAT = (TYPE = JSON COMPRESSION = NONE) SINGLE = TRUE OVERWRITE = TRUE MAX_FILE_SIZE = 100000000;

COPY INTO @RL_DATA/listings_v2_test.jsonl
FROM (SELECT OBJECT_CONSTRUCT('listing_id', listing_id, 'title', title, 'description', description_redacted,
        'price', price, 'template_id', template_id,
        'label', OBJECT_CONSTRUCT('category', gold_category, 'brand', gold_brand, 'violation', gold_violation))
      FROM LISTINGS_V2_REDACTED WHERE split = 'test')
FILE_FORMAT = (TYPE = JSON COMPRESSION = NONE) SINGLE = TRUE OVERWRITE = TRUE MAX_FILE_SIZE = 100000000;
