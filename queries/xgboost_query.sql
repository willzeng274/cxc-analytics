SELECT 
    c.company_id,
    c.age,
    DENSE_RANK() OVER (ORDER BY c.ecosystem_name) - 1 as ecosystem_id,
    DENSE_RANK() OVER (ORDER BY c.primary_tag) - 1 as primary_tag_id,
    COALESCE(SUM(d.amount), 0) as total_funding
FROM companies c
LEFT JOIN deals d ON c.company_id = d.company_id
GROUP BY 
    c.company_id,
    c.age,
    c.ecosystem_name,
    c.primary_tag; 