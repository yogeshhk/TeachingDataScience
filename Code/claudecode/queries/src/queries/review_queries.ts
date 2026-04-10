import { Database } from "sqlite";

interface Review {
  review_id: number;
  rating: number;
  review_text: string;
  review_date: string;
  [key: string]: any;
}

export async function getProductReviews(
  db: Database,
  productId: number,
  limit: number = 50
): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        c.first_name || ' ' || c.last_name as customer_name,
        c.customer_id,
        o.order_date,
        CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END as verified_purchase,
        (SELECT COUNT(*) FROM reviews r2 WHERE r2.customer_id = c.customer_id) as customer_review_count
    FROM reviews r
    JOIN customers c ON r.customer_id = c.customer_id
    LEFT JOIN order_items oi ON oi.product_id = r.product_id AND oi.order_id = r.order_id
    LEFT JOIN orders o ON o.order_id = r.order_id
    WHERE r.product_id = ?
    ORDER BY r.review_date DESC
    LIMIT ?
  `;

  const rows = await db.all(query, [productId, limit]);
  return rows as Review[];
}

export async function fetchCustomerReviews(
  db: Database,
  customerId: number
): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        p.name as product_name,
        p.product_id,
        r.order_id as order_number,
        (SELECT AVG(o.total_amount) 
         FROM orders o 
         WHERE o.customer_id = ?) as customer_avg_order_value
    FROM reviews r
    JOIN products p ON r.product_id = p.product_id
    WHERE r.customer_id = ?
    ORDER BY r.review_date DESC
  `;

  const rows = await db.all(query, [customerId, customerId]);
  return rows as Review[];
}

export async function findUnverifiedReviews(db: Database): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        c.email as customer_email,
        c.customer_id,
        JULIANDAY('now') - JULIANDAY(c.created_at) as account_age_days,
        (SELECT GROUP_CONCAT(DISTINCT p2.name, ', ')
         FROM orders o2
         JOIN order_items oi2 ON o2.order_id = oi2.order_id
         JOIN products p2 ON oi2.product_id = p2.product_id
         WHERE o2.customer_id = c.customer_id
         AND p2.product_id != r.product_id
         LIMIT 5) as other_products_bought
    FROM reviews r
    JOIN customers c ON r.customer_id = c.customer_id
    WHERE r.order_id IS NULL 
       OR NOT EXISTS (
           SELECT 1 FROM orders o 
           WHERE o.order_id = r.order_id 
           AND o.customer_id = r.customer_id
       )
    ORDER BY r.review_date DESC
  `;

  const rows = await db.all(query, []);
  return rows as Review[];
}

export async function getHelpfulReviews(
  db: Database,
  minHelpful: number = 5
): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        r.helpful_count,
        c.customer_id,
        c.first_name || ' ' || c.last_name as customer_name,
        c.email,
        c.created_at as customer_since,
        p.name as product_name,
        cat.name as product_category,
        CASE 
            WHEN COUNT(DISTINCT o.order_id) >= 10 THEN 'VIP'
            WHEN COUNT(DISTINCT o.order_id) >= 5 THEN 'Regular'
            WHEN COUNT(DISTINCT o.order_id) >= 2 THEN 'Returning'
            ELSE 'New'
        END as customer_segment
    FROM reviews r
    JOIN customers c ON r.customer_id = c.customer_id
    JOIN products p ON r.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    WHERE r.helpful_count >= ?
    GROUP BY r.review_id
    ORDER BY r.helpful_count DESC, r.review_date DESC
  `;

  const rows = await db.all(query, [minHelpful]);
  return rows as Review[];
}

export async function fetchRecentReviews(
  db: Database,
  days: number = 7
): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        p.name as product_name,
        p.product_id,
        COALESCE(SUM(i.quantity), 0) as product_inventory_status,
        c.city as customer_city,
        c.state as customer_state,
        o.total_amount as order_total
    FROM reviews r
    JOIN customers c ON r.customer_id = c.customer_id
    JOIN products p ON r.product_id = p.product_id
    LEFT JOIN inventory i ON i.product_id = p.product_id
    LEFT JOIN orders o ON o.order_id = r.order_id
    WHERE r.review_date >= date('now', '-' || ? || ' days')
    GROUP BY r.review_id
    ORDER BY r.review_date DESC
  `;

  const rows = await db.all(query, [days]);
  return rows as Review[];
}

export async function getReviewsByRating(
  db: Database,
  rating: number
): Promise<Review[]> {
  const query = `
    SELECT 
        r.review_id,
        r.rating,
        r.review_text,
        r.review_date,
        p.name as product_name,
        p.price as product_price,
        c.first_name || ' ' || c.last_name as customer_name,
        (SELECT COUNT(*) FROM orders WHERE customer_id = c.customer_id) as customer_order_history_count,
        CASE 
            WHEN r.order_id IS NOT NULL THEN 
                JULIANDAY(r.review_date) - JULIANDAY(o.order_date)
            ELSE NULL
        END as days_since_purchase
    FROM reviews r
    JOIN customers c ON r.customer_id = c.customer_id
    JOIN products p ON r.product_id = p.product_id
    LEFT JOIN orders o ON o.order_id = r.order_id
    WHERE r.rating = ?
    ORDER BY r.review_date DESC
  `;

  const rows = await db.all(query, [rating]);
  return rows as Review[];
}
