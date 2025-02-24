WITH monthly_amounts AS (
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(amount) as monthly_amount
    FROM deals
    GROUP BY DATE_TRUNC('month', date)
    ORDER BY month
),
numbers AS (
    SELECT generate_series(1, (
        SELECT COUNT(*) - 12 FROM monthly_amounts
    )) as seq_num
)
SELECT 
    array_agg(ma.monthly_amount) OVER (
        ORDER BY ma.month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ) as input_sequence,
    LEAD(ma.monthly_amount, 1) OVER (ORDER BY ma.month) as target_value
FROM monthly_amounts ma
WHERE EXISTS (
    SELECT 1 
    FROM monthly_amounts ma2
    WHERE ma2.month >= ma.month + interval '1 month'
); 