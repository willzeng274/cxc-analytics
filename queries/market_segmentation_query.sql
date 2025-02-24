SELECT 
    c.company_id,
    c.age,
    DENSE_RANK() OVER (ORDER BY c.ecosystem_name) - 1 as ecosystem_id,
    DENSE_RANK() OVER (ORDER BY c.primary_tag) - 1 as primary_tag_id
FROM companies c; 