WITH company_funding AS (
    SELECT 
        c.company_id,
        c.age,
        c.ecosystem_name,
        c.primary_tag,
        COALESCE(SUM(d.amount), 0) as total_funding
    FROM companies c
    LEFT JOIN deals d ON c.company_id = d.company_id
    GROUP BY c.company_id, c.age, c.ecosystem_name, c.primary_tag
),
funding_stats AS (
    SELECT AVG(total_funding) as median_funding
    FROM (
        SELECT total_funding
        FROM company_funding
        ORDER BY total_funding DESC
        LIMIT 1
    ) t
)
SELECT 
    cf.*,
    DENSE_RANK() OVER (ORDER BY ecosystem_name) - 1 as ecosystem_id,
    DENSE_RANK() OVER (ORDER BY primary_tag) - 1 as primary_tag_id,
    CASE WHEN cf.total_funding > fs.median_funding THEN 1 ELSE 0 END as is_successful
FROM company_funding cf
CROSS JOIN funding_stats fs;