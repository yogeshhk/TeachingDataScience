import { Database } from "sqlite";

interface Promotion {
  promotion_id: number;
  code: string;
  description: string;
  discount_percentage?: number;
  discount_amount?: number;
  min_order_value?: number;
  start_date: string;
  end_date: string;
  max_uses?: number;
  current_uses: number;
  usage_percentage?: number;
  times_used?: number;
  unique_customers?: number;
  total_discount_given?: number;
  avg_order_value?: number;
  top_products?: string;
  customer_segments?: string;
}

interface PromoEligibility {
  promotion_id?: number;
  code?: string;
  description?: string;
  discount_percentage?: number;
  discount_amount?: number;
  min_order_value?: number;
  start_date?: string;
  end_date?: string;
  max_uses?: number;
  current_uses?: number;
  max_uses_per_customer?: number;
  is_active?: number;
  eligibility_status: string;
  customer_use_count?: number;
  total_orders?: number;
  lifetime_value?: number;
  avg_order_value?: number;
  last_order_date?: string;
  current_cart_value?: number;
  cart_item_count?: number;
  past_promotions?: string;
  promo_code?: string;
}

interface ExpiringPromotion {
  promotion_id: number;
  code: string;
  description: string;
  discount_percentage?: number;
  discount_amount?: number;
  end_date: string;
  max_uses?: number;
  current_uses: number;
  days_until_expiry: number;
  usage_rate: number;
  total_orders: number;
  revenue_generated: number;
  total_discount_given: number;
  net_revenue: number;
  top_customer_segment: string;
}

interface PromotionPerformance {
  promotion_id: number;
  code: string;
  description: string;
  discount_percentage?: number;
  discount_amount?: number;
  start_date: string;
  end_date: string;
  max_uses?: number;
  current_uses: number;
  is_active: number;
  total_orders: number;
  unique_customers: number;
  gross_revenue: number;
  total_discount: number;
  net_revenue: number;
  avg_order_value: number;
  first_use_date: string;
  last_use_date: string;
  top_products: string;
  customer_breakdown: string;
  avg_promo_order: number;
  avg_regular_order: number;
  order_value_increase_percent: number;
}

interface UnusedPromotion {
  promotion_id: number;
  code: string;
  description: string;
  discount_percentage?: number;
  discount_amount?: number;
  min_order_value?: number;
  start_date: string;
  end_date: string;
  is_active: number;
  status: string;
  eligible_customer_count: number;
  most_similar_successful: string;
  target_segment: string;
  target_segment_size: number;
}

export async function getActivePromotions(db: Database): Promise<Promotion[]> {
  const query = `
    WITH active_promos AS (
        SELECT
            p.promotion_id,
            p.code,
            p.description,
            p.discount_percentage,
            p.discount_amount,
            p.min_order_value,
            p.start_date,
            p.end_date,
            p.max_uses,
            p.current_uses,
            CASE
                WHEN p.max_uses > 0 THEN ROUND(CAST(p.current_uses AS FLOAT) / p.max_uses * 100, 2)
                ELSE 0
            END as usage_percentage
        FROM promotions p
        WHERE p.is_active = 1
            AND datetime('now') BETWEEN p.start_date AND p.end_date
    ),
    promo_usage_stats AS (
        SELECT
            op.promotion_id,
            COUNT(DISTINCT o.order_id) as times_used,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            SUM(op.discount_applied) as total_discount_given,
            AVG(o.total_amount) as avg_order_value
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        GROUP BY op.promotion_id
    ),
    top_products_with_promo AS (
        SELECT
            op.promotion_id,
            p.name as product_name,
            p.category_id,
            c.name as category_name,
            COUNT(DISTINCT oi.order_id) as purchase_count,
            ROW_NUMBER() OVER (PARTITION BY op.promotion_id ORDER BY COUNT(DISTINCT oi.order_id) DESC) as rn
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        GROUP BY op.promotion_id, p.product_id
    ),
    customer_segments AS (
        SELECT
            op.promotion_id,
            CASE
                WHEN COUNT(DISTINCT o2.order_id) >= 5 THEN 'Loyal'
                WHEN COUNT(DISTINCT o2.order_id) >= 2 THEN 'Returning'
                ELSE 'New'
            END as customer_segment,
            COUNT(DISTINCT o.customer_id) as segment_count
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        LEFT JOIN orders o2 ON o.customer_id = o2.customer_id AND o2.order_date < o.order_date
        GROUP BY op.promotion_id, customer_segment
    )
    SELECT
        ap.*,
        COALESCE(pus.times_used, 0) as times_used,
        COALESCE(pus.unique_customers, 0) as unique_customers,
        COALESCE(pus.total_discount_given, 0) as total_discount_given,
        COALESCE(pus.avg_order_value, 0) as avg_order_value,
        GROUP_CONCAT(
            CASE WHEN tpp.rn <= 3 THEN tpp.product_name || ' (' || tpp.category_name || ')' END,
            ', '
        ) as top_products,
        GROUP_CONCAT(
            cs.customer_segment || ': ' || cs.segment_count,
            ', '
        ) as customer_segments
    FROM active_promos ap
    LEFT JOIN promo_usage_stats pus ON ap.promotion_id = pus.promotion_id
    LEFT JOIN top_products_with_promo tpp ON ap.promotion_id = tpp.promotion_id AND tpp.rn <= 3
    LEFT JOIN customer_segments cs ON ap.promotion_id = cs.promotion_id
    GROUP BY ap.promotion_id
    ORDER BY ap.usage_percentage DESC, pus.times_used DESC
    `;

  const rows = await db.all(query);
  return rows as Promotion[];
}

export async function checkPromoEligibility(
  db: Database,
  customerId: number,
  promoCode: string
): Promise<PromoEligibility> {
  const query = `
    WITH promo_info AS (
        SELECT
            promotion_id,
            code,
            description,
            discount_percentage,
            discount_amount,
            min_order_value,
            start_date,
            end_date,
            max_uses,
            current_uses,
            max_uses_per_customer,
            is_active
        FROM promotions
        WHERE code = ?
    ),
    customer_promo_usage AS (
        SELECT
            COUNT(*) as customer_use_count
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        JOIN promo_info pi ON op.promotion_id = pi.promotion_id
        WHERE o.customer_id = ?
    ),
    customer_order_history AS (
        SELECT
            COUNT(*) as total_orders,
            SUM(total_amount) as lifetime_value,
            AVG(total_amount) as avg_order_value,
            MAX(order_date) as last_order_date
        FROM orders
        WHERE customer_id = ?
    ),
    current_cart AS (
        SELECT
            SUM(ci.quantity * p.price) as cart_value,
            COUNT(DISTINCT ci.product_id) as item_count
        FROM cart_items ci
        JOIN products p ON ci.product_id = p.product_id
        WHERE ci.customer_id = ?
    ),
    past_promo_usage AS (
        SELECT
            p.code,
            p.description,
            COUNT(*) as times_used,
            MAX(o.order_date) as last_used,
            SUM(op.discount_applied) as total_saved
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        JOIN promotions p ON op.promotion_id = p.promotion_id
        WHERE o.customer_id = ?
        GROUP BY p.promotion_id
        ORDER BY last_used DESC
        LIMIT 5
    )
    SELECT
        pi.*,
        CASE
            WHEN pi.promotion_id IS NULL THEN 'Invalid promotion code'
            WHEN pi.is_active = 0 THEN 'Promotion is not active'
            WHEN datetime('now') < pi.start_date THEN 'Promotion has not started yet'
            WHEN datetime('now') > pi.end_date THEN 'Promotion has expired'
            WHEN pi.max_uses > 0 AND pi.current_uses >= pi.max_uses THEN 'Promotion usage limit reached'
            WHEN pi.max_uses_per_customer > 0 AND cpu.customer_use_count >= pi.max_uses_per_customer THEN 'Customer usage limit reached'
            WHEN cc.cart_value < pi.min_order_value THEN 'Cart value below minimum order requirement'
            ELSE 'Eligible'
        END as eligibility_status,
        cpu.customer_use_count,
        coh.total_orders,
        coh.lifetime_value,
        coh.avg_order_value,
        coh.last_order_date,
        cc.cart_value as current_cart_value,
        cc.item_count as cart_item_count,
        (SELECT GROUP_CONCAT(code || ' (used ' || times_used || 'x)', ', ') FROM past_promo_usage) as past_promotions
    FROM promo_info pi
    CROSS JOIN customer_promo_usage cpu
    CROSS JOIN customer_order_history coh
    CROSS JOIN current_cart cc
    `;

  const row = await db.get(query, [
    promoCode,
    customerId,
    customerId,
    customerId,
    customerId,
  ]);
  if (row) {
    return row as PromoEligibility;
  } else {
    return {
      eligibility_status: "Invalid promotion code",
      promo_code: promoCode,
    };
  }
}

export async function findExpiringPromotions(
  db: Database,
  days: number = 7
): Promise<ExpiringPromotion[]> {
  const expiryDate = new Date();
  expiryDate.setDate(expiryDate.getDate() + days);
  const expiryDateStr = expiryDate.toISOString().replace("T", " ").slice(0, 19);

  const query = `
    WITH expiring_promos AS (
        SELECT
            p.promotion_id,
            p.code,
            p.description,
            p.discount_percentage,
            p.discount_amount,
            p.end_date,
            p.max_uses,
            p.current_uses,
            CAST((julianday(p.end_date) - julianday('now')) AS INTEGER) as days_until_expiry,
            CASE
                WHEN p.max_uses > 0 THEN ROUND(CAST(p.current_uses AS FLOAT) / p.max_uses * 100, 2)
                ELSE ROUND(CAST(p.current_uses AS FLOAT) / 100 * 100, 2)
            END as usage_rate
        FROM promotions p
        WHERE p.is_active = 1
            AND p.end_date BETWEEN datetime('now') AND ?
    ),
    promo_stats AS (
        SELECT
            op.promotion_id,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(o.total_amount) as revenue_generated,
            SUM(op.discount_applied) as total_discount_given,
            SUM(o.total_amount - op.discount_applied) as net_revenue
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        GROUP BY op.promotion_id
    ),
    customer_segments AS (
        SELECT
            op.promotion_id,
            CASE
                WHEN c.total_spent >= 1000 THEN 'VIP'
                WHEN c.total_spent >= 500 THEN 'Premium'
                WHEN c.total_spent >= 100 THEN 'Regular'
                ELSE 'New'
            END as segment,
            COUNT(DISTINCT o.customer_id) as customer_count
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY op.promotion_id, segment
    ),
    top_segment AS (
        SELECT
            promotion_id,
            segment as top_customer_segment,
            customer_count
        FROM customer_segments
        WHERE (promotion_id, customer_count) IN (
            SELECT promotion_id, MAX(customer_count)
            FROM customer_segments
            GROUP BY promotion_id
        )
    )
    SELECT
        ep.*,
        COALESCE(ps.total_orders, 0) as total_orders,
        COALESCE(ps.revenue_generated, 0) as revenue_generated,
        COALESCE(ps.total_discount_given, 0) as total_discount_given,
        COALESCE(ps.net_revenue, 0) as net_revenue,
        COALESCE(ts.top_customer_segment, 'None') as top_customer_segment
    FROM expiring_promos ep
    LEFT JOIN promo_stats ps ON ep.promotion_id = ps.promotion_id
    LEFT JOIN top_segment ts ON ep.promotion_id = ts.promotion_id
    ORDER BY ep.days_until_expiry ASC, ep.usage_rate DESC
    `;

  const rows = await db.all(query, [expiryDateStr]);
  return rows as ExpiringPromotion[];
}

export async function getPromotionPerformance(
  db: Database,
  promoId: number
): Promise<PromotionPerformance | null> {
  const query = `
    WITH promo_details AS (
        SELECT
            promotion_id,
            code,
            description,
            discount_percentage,
            discount_amount,
            start_date,
            end_date,
            max_uses,
            current_uses,
            is_active
        FROM promotions
        WHERE promotion_id = ?
    ),
    overall_stats AS (
        SELECT
            COUNT(DISTINCT o.order_id) as total_orders,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            SUM(o.total_amount) as gross_revenue,
            SUM(op.discount_applied) as total_discount,
            SUM(o.total_amount - op.discount_applied) as net_revenue,
            AVG(o.total_amount) as avg_order_value,
            MIN(o.order_date) as first_use_date,
            MAX(o.order_date) as last_use_date
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        WHERE op.promotion_id = ?
    ),
    products_bought AS (
        SELECT
            p.name as product_name,
            c.name as category_name,
            COUNT(DISTINCT oi.order_id) as times_purchased,
            SUM(oi.quantity) as total_quantity,
            SUM(oi.quantity * oi.price) as revenue
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        WHERE op.promotion_id = ?
        GROUP BY p.product_id
        ORDER BY times_purchased DESC
        LIMIT 10
    ),
    customer_analysis AS (
        SELECT
            CASE
                WHEN o2.customer_id IS NULL THEN 'New Customer'
                ELSE 'Returning Customer'
            END as customer_type,
            COUNT(DISTINCT o.customer_id) as customer_count,
            AVG(o.total_amount) as avg_order_value
        FROM order_promotions op
        JOIN orders o ON op.order_id = o.order_id
        LEFT JOIN (
            SELECT DISTINCT customer_id
            FROM orders
            WHERE order_date < (SELECT MIN(order_date) FROM orders WHERE order_id IN (SELECT order_id FROM order_promotions WHERE promotion_id = ?))
        ) o2 ON o.customer_id = o2.customer_id
        WHERE op.promotion_id = ?
        GROUP BY customer_type
    ),
    order_increase AS (
        SELECT
            AVG(CASE WHEN op.order_id IS NOT NULL THEN o.total_amount ELSE NULL END) as avg_promo_order,
            AVG(CASE WHEN op.order_id IS NULL THEN o.total_amount ELSE NULL END) as avg_regular_order
        FROM orders o
        LEFT JOIN order_promotions op ON o.order_id = op.order_id AND op.promotion_id = ?
        WHERE o.order_date BETWEEN (SELECT start_date FROM promo_details) AND (SELECT end_date FROM promo_details)
    )
    SELECT
        pd.*,
        os.*,
        (SELECT GROUP_CONCAT(product_name || ' (' || category_name || '): ' || times_purchased || ' orders', ', ') FROM products_bought) as top_products,
        (SELECT GROUP_CONCAT(customer_type || ': ' || customer_count, ', ') FROM customer_analysis) as customer_breakdown,
        oi.avg_promo_order,
        oi.avg_regular_order,
        CASE
            WHEN oi.avg_regular_order > 0 THEN ROUND((oi.avg_promo_order - oi.avg_regular_order) / oi.avg_regular_order * 100, 2)
            ELSE 0
        END as order_value_increase_percent
    FROM promo_details pd
    CROSS JOIN overall_stats os
    CROSS JOIN order_increase oi
    `;

  const row = await db.get(query, [
    promoId,
    promoId,
    promoId,
    promoId,
    promoId,
    promoId,
  ]);
  if (row) {
    return row as PromotionPerformance;
  } else {
    return null;
  }
}

export async function findUnusedPromotions(
  db: Database
): Promise<UnusedPromotion[]> {
  const query = `
    WITH unused_promos AS (
        SELECT
            p.promotion_id,
            p.code,
            p.description,
            p.discount_percentage,
            p.discount_amount,
            p.min_order_value,
            p.start_date,
            p.end_date,
            p.is_active,
            CASE
                WHEN datetime('now') < p.start_date THEN 'Not Started'
                WHEN datetime('now') > p.end_date THEN 'Expired'
                WHEN p.is_active = 0 THEN 'Inactive'
                ELSE 'Active but Unused'
            END as status
        FROM promotions p
        WHERE p.current_uses = 0
    ),
    eligible_customers AS (
        SELECT
            up.promotion_id,
            COUNT(DISTINCT c.customer_id) as eligible_customer_count
        FROM unused_promos up
        CROSS JOIN customers c
        WHERE c.total_spent >= up.min_order_value
            OR up.min_order_value = 0
        GROUP BY up.promotion_id
    ),
    similar_successful AS (
        SELECT
            up.promotion_id,
            p2.code as similar_promo_code,
            p2.description as similar_promo_desc,
            p2.current_uses as similar_promo_uses,
            CASE
                WHEN up.discount_percentage > 0 AND p2.discount_percentage > 0 THEN ABS(up.discount_percentage - p2.discount_percentage)
                WHEN up.discount_amount > 0 AND p2.discount_amount > 0 THEN ABS(up.discount_amount - p2.discount_amount)
                ELSE 999
            END as similarity_score
        FROM unused_promos up
        JOIN promotions p2 ON p2.promotion_id != up.promotion_id
        WHERE p2.current_uses > 10
            AND (
                (up.discount_percentage > 0 AND p2.discount_percentage > 0)
                OR (up.discount_amount > 0 AND p2.discount_amount > 0)
            )
        ORDER BY similarity_score ASC
    ),
    target_segment AS (
        SELECT
            up.promotion_id,
            CASE
                WHEN up.min_order_value >= 500 THEN 'VIP Customers'
                WHEN up.min_order_value >= 200 THEN 'Premium Customers'
                WHEN up.min_order_value >= 50 THEN 'Regular Customers'
                ELSE 'All Customers'
            END as target_segment,
            COUNT(DISTINCT c.customer_id) as segment_size
        FROM unused_promos up
        CROSS JOIN customers c
        WHERE (
            CASE
                WHEN up.min_order_value >= 500 THEN c.total_spent >= 1000
                WHEN up.min_order_value >= 200 THEN c.total_spent >= 500
                WHEN up.min_order_value >= 50 THEN c.total_spent >= 100
                ELSE 1
            END
        ) = 1
        GROUP BY up.promotion_id
    )
    SELECT
        up.*,
        ec.eligible_customer_count,
        COALESCE(
            (SELECT similar_promo_code || ' (' || similar_promo_uses || ' uses)'
             FROM similar_successful ss
             WHERE ss.promotion_id = up.promotion_id
             ORDER BY similarity_score ASC
             LIMIT 1),
            'No similar successful promotions'
        ) as most_similar_successful,
        ts.target_segment,
        ts.segment_size as target_segment_size
    FROM unused_promos up
    LEFT JOIN eligible_customers ec ON up.promotion_id = ec.promotion_id
    LEFT JOIN target_segment ts ON up.promotion_id = ts.promotion_id
    ORDER BY up.status, ec.eligible_customer_count DESC
    `;

  const rows = await db.all(query);
  return rows as UnusedPromotion[];
}
