
# return users where max(updated_date) per user != max(updated_date) all
query = """
WITH TABLE_X AS (
	SELECT t.user_id, MAX(t.date)
	FROM public.tracking t
	GROUP BY t.user_id
)

SELECT t.user_id, t.date
FROM public.TABLE_X t
JOIN TABLE_X latest_dates
ON t.user_id = latest_dates.user_id AND t.date = latest_dates.latest_date
WHERE t.date != MAX(latest_dates.latest_date)
LIMIT 20;
"""