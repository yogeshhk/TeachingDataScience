import { Database } from "sqlite";

interface OrderItem {
  order_item_id: number;
  product_id: number;
  product_name: string;
  quantity: number;
  price_at_time: number;
}

interface OrderDetails {
  order_id: number;
  order_date: string;
  status: string;
  total_amount: number;
  customer_email: string;
  shipping_address: string;
  shipping_city: string;
  shipping_state: string;
  shipping_zip: string;
  items: OrderItem[];
}

export async function getOrderDetails(
  db: Database,
  orderId: number
): Promise<OrderDetails | null> {
  const query = `
    SELECT 
        o.order_id,
        o.order_date,
        o.status,
        o.total_amount,
        c.email as customer_email,
        o.shipping_address,
        o.shipping_city,
        o.shipping_state,
        o.shipping_zip,
        oi.order_item_id,
        oi.product_id,
        p.name as product_name,
        oi.quantity,
        oi.price_at_time
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_id = ?
    `;

  const rows: any[] = await db.all(query, [orderId]);

  if (!rows || rows.length === 0) {
    return null;
  }

  const order: OrderDetails = {
    order_id: rows[0].order_id,
    order_date: rows[0].order_date,
    status: rows[0].status,
    total_amount: rows[0].total_amount,
    customer_email: rows[0].customer_email,
    shipping_address: rows[0].shipping_address,
    shipping_city: rows[0].shipping_city,
    shipping_state: rows[0].shipping_state,
    shipping_zip: rows[0].shipping_zip,
    items: [],
  };

  for (const row of rows) {
    order.items.push({
      order_item_id: row.order_item_id,
      product_id: row.product_id,
      product_name: row.product_name,
      quantity: row.quantity,
      price_at_time: row.price_at_time,
    });
  }

  return order;
}

export async function fetchCustomerOrders(
  db: Database,
  customerId: number,
  limit: number = 10
): Promise<any[]> {
  const query = `
    SELECT 
        o.order_id,
        o.order_date,
        o.status,
        o.total_amount,
        o.shipping_city,
        o.shipping_state,
        COUNT(oi.order_item_id) as item_count
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.customer_id = ?
    GROUP BY o.order_id, o.order_date, o.status, o.total_amount, 
             o.shipping_city, o.shipping_state
    ORDER BY o.order_date DESC
    LIMIT ?
    `;

  const rows = await db.all(query, [customerId, limit]);
  return rows;
}

export async function getPendingOrders(db: Database): Promise<any[]> {
  const query = `
    SELECT 
        o.order_id,
        o.order_date,
        o.total_amount,
        c.first_name || ' ' || c.last_name as customer_name,
        c.phone,
        julianday('now') - julianday(o.order_date) as days_since_created
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'pending'
    ORDER BY o.order_date
    `;

  const rows = await db.all(query, []);
  return rows;
}

export async function findOrdersByStatus(
  db: Database,
  status: string
): Promise<any[]> {
  const query = `
    SELECT DISTINCT
        o.order_id,
        o.order_date,
        o.total_amount,
        c.email as customer_email,
        GROUP_CONCAT(DISTINCT p.sku) as product_skus,
        GROUP_CONCAT(DISTINCT w.name) as warehouses
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN inventory i ON p.product_id = i.product_id
    LEFT JOIN warehouses w ON i.warehouse_id = w.warehouse_id
    WHERE o.status = ?
    GROUP BY o.order_id, o.order_date, o.total_amount, c.email
    ORDER BY o.order_date DESC
    `;

  const rows = await db.all(query, [status]);
  return rows;
}

export async function getRecentOrders(
  db: Database,
  days: number = 7
): Promise<any[]> {
  const query = `
    SELECT DISTINCT
        o.order_id,
        o.order_date,
        o.total_amount,
        o.shipping_amount,
        c.segment as customer_segment,
        CASE 
            WHEN o.shipping_amount = 0 THEN 'Free Shipping'
            WHEN o.shipping_amount < 10 THEN 'Standard'
            WHEN o.shipping_amount < 25 THEN 'Express'
            ELSE 'Priority'
        END as shipping_method,
        GROUP_CONCAT(DISTINCT cat.name) as product_categories
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN categories cat ON p.category_id = cat.category_id
    WHERE o.order_date >= date('now', '-' || ? || ' days')
    GROUP BY o.order_id, o.order_date, o.total_amount, o.shipping_amount, c.segment
    ORDER BY o.order_date DESC
    `;

  const rows = await db.all(query, [days]);
  return rows;
}

export async function fetchOrdersByDateRange(
  db: Database,
  startDate: string,
  endDate: string
): Promise<any[]> {
  const query = `
    SELECT 
        o.order_id,
        o.order_date,
        o.total_amount,
        c.status as customer_status,
        o.billing_state,
        COUNT(oi.order_item_id) as item_count
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_date >= ? AND o.order_date <= ?
    GROUP BY o.order_id, o.order_date, o.total_amount, c.status, o.billing_state
    ORDER BY o.order_date DESC
    `;

  const rows = await db.all(query, [startDate, endDate]);
  return rows;
}

export async function getHighValueOrders(
  db: Database,
  minAmount: number = 500
): Promise<any[]> {
  const query = `
    WITH customer_ltv AS (
        SELECT 
            customer_id,
            SUM(total_amount) as lifetime_value
        FROM orders
        GROUP BY customer_id
    )
    SELECT DISTINCT
        o.order_id,
        o.order_date,
        o.total_amount,
        c.email,
        ltv.lifetime_value as customer_lifetime_value,
        o.shipping_address,
        o.shipping_city,
        o.shipping_state,
        o.shipping_zip,
        GROUP_CONCAT(p.name) as product_names
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN customer_ltv ltv ON c.customer_id = ltv.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE o.total_amount >= ?
    GROUP BY o.order_id, o.order_date, o.total_amount, c.email, 
             ltv.lifetime_value, o.shipping_address, o.shipping_city, 
             o.shipping_state, o.shipping_zip
    ORDER BY o.total_amount DESC
    `;

  const rows = await db.all(query, [minAmount]);
  return rows;
}
