SELECT 
    DATE_TRUNC('month', date) as month,
    SUM(amount) as monthly_amount
FROM deals
GROUP BY DATE_TRUNC('month', date)
ORDER BY month; 