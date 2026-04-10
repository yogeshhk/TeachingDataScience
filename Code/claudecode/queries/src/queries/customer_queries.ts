import { Database } from "sqlite";

interface Address {
  address_id: number;
  customer_id: number;
  street: string;
  city: string;
  state: string;
  zip_code: string;
  country: string;
  is_default: number;
  [key: string]: any;
}

export async function getCustomerByEmail(
  db: Database,
  email: string
): Promise<any> {
  const query = `
    SELECT 
        c.*,
        sa.street AS shipping_street,
        sa.city AS shipping_city,
        sa.state AS shipping_state,
        sa.zip_code AS shipping_zip,
        sa.country AS shipping_country,
        MAX(o.order_date) as last_order_date
    FROM customers c
    LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id AND sa.is_default = 1
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE c.email = ?
    GROUP BY c.customer_id
  `;

  // db.get returns a Promise directly
  return await db.get(query, [email]);
}

export async function fetchActiveCustomers(
  db: Database,
  daysInactive: number = 90
): Promise<any[]> {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysInactive);
  const cutoffDateStr = cutoffDate.toISOString().split("T")[0];

  const query = `
    SELECT 
        c.*,
        COUNT(DISTINCT o.order_id) as total_order_count,
        AVG(o.total_amount) as average_order_value
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= ?
    GROUP BY c.customer_id
  `;

  return await db.all(query, [cutoffDateStr]);
}

export async function findCustomersBySegment(
  db: Database,
  segmentName: string
): Promise<any[]> {
  const query = `
    SELECT 
        c.customer_id,
        c.email,
        c.phone,
        c.first_name,
        c.last_name,
        COALESCE(SUM(o.total_amount), 0) as total_lifetime_value
    FROM customers c
    INNER JOIN customer_segments cs ON c.customer_id = cs.customer_id
    INNER JOIN segments s ON cs.segment_id = s.segment_id
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE s.segment_name = ?
    GROUP BY c.customer_id
  `;

  return await db.all(query, [segmentName]);
}

export async function getCustomerProfile(
  db: Database,
  customerId: number
): Promise<any | null> {
  // Get customer basic info
  const customerQuery = "SELECT * FROM customers WHERE customer_id = ?";

  // Await the promise returned by db.get
  const customer = await db.get(customerQuery, [customerId]);

  if (!customer) {
    return null;
  }

  // Get all addresses
  const addressesQuery = `
    SELECT * FROM shipping_addresses 
    WHERE customer_id = ?
    ORDER BY is_default DESC, address_id
  `;

  // Await the promise returned by db.all
  const addresses = (await db.all(addressesQuery, [customerId])) as Address[];

  // Get order count
  const orderCountQuery =
    "SELECT COUNT(*) as order_count FROM orders WHERE customer_id = ?";

  // Await the promise returned by db.get
  const orderCountResult = await db.get(orderCountQuery, [customerId]);

  const orderCount = orderCountResult.order_count;

  // Get last 5 product names ordered
  const productsQuery = `
    SELECT DISTINCT p.product_name
    FROM orders o
    INNER JOIN order_items oi ON o.order_id = oi.order_id
    INNER JOIN products p ON oi.product_id = p.product_id
    WHERE o.customer_id = ?
    ORDER BY o.order_date DESC
    LIMIT 5
  `;

  // Await the promise returned by db.all
  const productRows = await db.all(productsQuery, [customerId]);

  const lastProducts = productRows.map((row) => row.product_name);

  // Combine results
  const result = {
    ...customer,
    addresses,
    order_count: orderCount,
    last_5_products: lastProducts,
  };

  return result;
}

export async function searchCustomersByName(
  db: Database,
  firstName?: string,
  lastName?: string
): Promise<any[]> {
  let query = `
    SELECT 
        c.*,
        c.status,
        CAST((julianday('now') - julianday(c.created_at)) AS INTEGER) as account_age_days,
        sa.city as default_city,
        sa.state as default_state
    FROM customers c
    LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id AND sa.is_default = 1
    WHERE 1=1
  `;

  const params: string[] = [];

  if (firstName) {
    query += " AND c.first_name LIKE ?";
    params.push(`%${firstName}%`);
  }

  if (lastName) {
    query += " AND c.last_name LIKE ?";
    params.push(`%${lastName}%`);
  }

  return await db.all(query, params);
}

export async function listCustomersWithReviews(db: Database): Promise<any[]> {
  const query = `
    SELECT 
        c.*,
        COUNT(r.review_id) as review_count,
        AVG(r.rating) as average_rating_given
    FROM customers c
    INNER JOIN reviews r ON c.customer_id = r.customer_id
    GROUP BY c.customer_id
    HAVING COUNT(r.review_id) > 0
  `;

  return await db.all(query);
}
