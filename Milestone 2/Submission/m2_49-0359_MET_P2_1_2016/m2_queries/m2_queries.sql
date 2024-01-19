-- Query1: All trip info(location,tip amount,etc) for the 20 highest trip distances.
SELECT 
	gt.*,
    pl."Original Value" AS pickup_location,
    dl."Original Value" AS dropoff_location
FROM 
    green_taxi_01_2016 gt
 JOIN 
    lookup_green_taxi_01_2016 pl ON CAST(gt.pickup_location AS text) = pl."Imputed/Encoded Value" AND pl."Column Name" = 'pickup_location'
 JOIN 
    lookup_green_taxi_01_2016 dl ON CAST(gt.dropoff_location AS text) = dl."Imputed/Encoded Value" AND dl."Column Name" = 'dropoff_location'
ORDER BY 
    gt.trip_distance DESC
LIMIT 20;


-- Query 2: What is the average fare amount per payment type.
SELECT 
    CASE
        WHEN "payment_type_Credit card" = 1.0 THEN 'Credit Card'
        WHEN "payment_type_Dispute" = 1.0 THEN 'Dispute'
        WHEN "payment_type_No charge" = 1.0 THEN 'No Charge'
        ELSE 'Cash'
    END AS payment_type,
    AVG(fare_amount) AS average_fare
FROM 
    green_taxi_01_2016
GROUP BY 
    CASE
        WHEN "payment_type_Credit card" = 1.0 THEN 'Credit Card'
        WHEN "payment_type_Dispute" = 1.0 THEN 'Dispute'
        WHEN "payment_type_No charge" = 1.0 THEN 'No Charge'
        ELSE 'Cash'
    END;

-- Query3: On average, which city tips the most.
CREATE VIEW city_lookup_view AS
SELECT 
    "Imputed/Encoded Value"::integer AS encoded_value,
    CASE 
        WHEN  "Column Name" = 'pickup_location' THEN SPLIT_PART("Original Value", ',', 1) 
    END AS city_name
FROM lookup_green_taxi_01_2016
WHERE  "Column Name" = 'pickup_location';

SELECT clv.city_name, AVG(gt.tip_amount) AS average_tip
FROM green_taxi_01_2016 gt
JOIN city_lookup_view clv ON gt.pickup_location = clv.encoded_value
GROUP BY clv.city_name
ORDER BY average_tip DESC
LIMIT 1;


--Query4: On average, which city tips the least.
CREATE VIEW city_lookup_view2 AS
SELECT 
    "Imputed/Encoded Value"::integer AS encoded_value,
    CASE 
        WHEN "Column Name" = 'pickup_location' THEN SPLIT_PART("Original Value", ',', 1) 
    END AS city_name
FROM lookup_green_taxi_01_2016
WHERE "Column Name" = 'pickup_location' ;

SELECT clv.city_name, AVG(gt.tip_amount) AS average_tip
FROM green_taxi_01_2016 gt
JOIN city_lookup_view clv ON gt.pickup_location = clv.encoded_value
GROUP BY clv.city_name
ORDER BY average_tip ASC
LIMIT 1;


--Query5: What is the most frequent destination on the weekend.
CREATE VIEW location_lookup_view AS
SELECT 
    "Imputed/Encoded Value"::integer AS encoded_value,
    "Original Value" AS location
FROM lookup_green_taxi_01_2016
WHERE "Column Name" = 'dropoff_location';

SELECT llv.location, COUNT(*) AS trip_count
FROM green_taxi_01_2016 tt
JOIN location_lookup_view llv ON tt.dropoff_location = llv.encoded_value
WHERE EXTRACT(DOW FROM tt.dropoff_datetime::timestamp) IN (6, 0) -- 6 = Saturday, 0 = Sunday
GROUP BY llv.location
ORDER BY trip_count DESC
LIMIT 1;

--Query6: On average which trip type travels longer distances
SELECT
  CASE
    WHEN AVG(CASE WHEN "trip_type_Street-hail" = 1.0 THEN trip_distance ELSE NULL END) >
         AVG(CASE WHEN "trip_type_Street-hail" = 0.0 THEN trip_distance ELSE NULL END)
    THEN 'Street-hail'
    ELSE 'Dispute'
  END AS trip_type_with_longer_average_distance,
  GREATEST(
    AVG(CASE WHEN "trip_type_Street-hail" = 1.0 THEN trip_distance ELSE NULL END),
    AVG(CASE WHEN "trip_type_Street-hail" = 0.0 THEN trip_distance ELSE NULL END)
  ) AS longer_average_distance
FROM 
  green_taxi_01_2016;

--Query7: between 4pm and 6pm what is the average fare amount.
SELECT AVG(fare_amount) AS average_fare
FROM green_taxi_01_2016
WHERE 
  EXTRACT(HOUR FROM CAST(pickup_datetime AS timestamp)) >= 16 
  AND EXTRACT(HOUR FROM CAST(pickup_datetime AS timestamp)) < 18
  AND EXTRACT(HOUR FROM CAST(dropoff_datetime AS timestamp)) >= 16
  AND (
    EXTRACT(HOUR FROM CAST(dropoff_datetime AS timestamp)) < 18 
    OR (
      EXTRACT(HOUR FROM CAST(dropoff_datetime AS timestamp)) = 18 
      AND EXTRACT(MINUTE FROM CAST(dropoff_datetime AS timestamp)) = 0 
      AND EXTRACT(SECOND FROM CAST(dropoff_datetime AS timestamp)) = 0
    )
  );

