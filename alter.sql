-- Active: 1772625852827@@localhost@5432@lab5_reddit@public
alter table posts add column keywords TEXT;
alter table posts add column distance_to_centroid FLOAT;