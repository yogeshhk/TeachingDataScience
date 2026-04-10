import { Database } from "sqlite";

interface ShippingAddress {
  address_id: number;
  street: string;
  city: string;
  state: string;
  zip_code: string;
  is_default: number;
  order_count: number;
  last_used_date: string | null;
}

interface OrderByDestination {
  order_id: number;
  order_date: string;
  status: string;
  customer_email: string;
  destination_city: string;
  destination_state: string;
  products_ordered: string;
  warehouse_name: string | null;
  warehouse_city: string | null;
  delivery_days: number | null;
}

interface UnshippedOrder {
  order_id: number;
  order_date: string;
  email: string;
  phone: string;
  previous_order_count: number;
  shipping_address: string;
  total_amount: number;
  inventory_status: string;
}

interface ShippingCostByState {
  state: string;
  order_count: number;
  total_shipping_cost: number;
  avg_shipping_cost: number;
  avg_order_weight: number;
  top_products: string;
}

interface DeliveryDelay {
  order_id: number;
  order_date: string;
  current_status: string;
  days_since_order: number;
  email: string;
  phone: string;
  customer_segment: string;
  destination: string;
  products: string;
  total_amount: number;
  last_known_warehouse: string | null;
}

export async function getShippingAddresses(
  db: Database,
  customerId: number
): Promise<ShippingAddress[]> {
  const query = `
    SELECT 
        sa.address_id,
        sa.street,
        sa.city,
        sa.state,
        sa.zip_code,
        sa.is_default,
        COUNT(DISTINCT o.order_id) as order_count,
        MAX(o.created_at) as last_used_date
    FROM shipping_addresses sa
    LEFT JOIN orders o ON sa.address_id = o.shipping_address_id
    WHERE sa.customer_id = ?
    GROUP BY sa.address_id, sa.street, sa.city, sa.state, sa.zip_code, sa.is_default
    ORDER BY sa.is_default DESC, order_count DESC
    `;

  const rows: any[] = await db.all(query, [customerId]);
  return rows as ShippingAddress[];
}

export async function findOrdersByDestination(
  db: Database,
  state: string
): Promise<OrderByDestination[]> {
  const query = `
    SELECT 
        o.order_id,
        o.created_at as order_date,
        o.status,
        c.email as customer_email,
        sa.city as destination_city,
        sa.state as destination_state,
        GROUP_CONCAT(p.name || ' (x' || oi.quantity || ')', ', ') as products_ordered,
        w.name as warehouse_name,
        w.city as warehouse_city,
        CASE 
            WHEN o.status = 'delivered' THEN 
                CAST((julianday(o.updated_at) - julianday(o.created_at)) AS INTEGER)
            ELSE NULL
        END as delivery_days
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN shipping_addresses sa ON o.shipping_address_id = sa.address_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN warehouses w ON o.warehouse_id = w.warehouse_id
    WHERE sa.state = ?
    GROUP BY o.order_id, o.created_at, o.status, c.email, 
             sa.city, sa.state, w.name, w.city, o.updated_at
    ORDER BY o.created_at DESC
    `;

  const rows: any[] = await db.all(query, [state]);
  return rows as OrderByDestination[];
}

export async function getUnshippedOrders(
  db: Database
): Promise<UnshippedOrder[]> {
  const query = `
    WITH customer_order_counts AS (
        SELECT 
            customer_id,
            COUNT(*) as total_orders
        FROM orders
        WHERE status = 'delivered'
        GROUP BY customer_id
    ),
    inventory_check AS (
        SELECT 
            oi.order_id,
            MIN(CASE 
                WHEN COALESCE(inv.quantity, 0) >= oi.quantity THEN 1 
                ELSE 0 
            END) as all_items_available
        FROM order_items oi
        LEFT JOIN inventory inv ON oi.product_id = inv.product_id
        GROUP BY oi.order_id
    )
    SELECT 
        o.order_id,
        o.created_at as order_date,
        c.email,
        c.phone,
        COALESCE(coc.total_orders, 0) as previous_order_count,
        sa.street || ', ' || sa.city || ', ' || sa.state || ' ' || sa.zip_code as shipping_address,
        o.total_amount,
        CASE 
            WHEN ic.all_items_available = 1 THEN 'Ready to ship'
            ELSE 'Inventory shortage'
        END as inventory_status
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN shipping_addresses sa ON o.shipping_address_id = sa.address_id
    LEFT JOIN customer_order_counts coc ON c.customer_id = coc.customer_id
    LEFT JOIN inventory_check ic ON o.order_id = ic.order_id
    WHERE o.status IN ('pending', 'processing')
    ORDER BY o.created_at ASC
    `;

  const rows: any[] = await db.all(query);
  return rows as UnshippedOrder[];
}

export async function calculateShippingCostsByState(
  db: Database
): Promise<ShippingCostByState[]> {
  const query = `
    WITH state_products AS (
        SELECT 
            sa.state,
            p.name as product_name,
            SUM(oi.quantity) as total_quantity,
            ROW_NUMBER() OVER (PARTITION BY sa.state ORDER BY SUM(oi.quantity) DESC) as rn
        FROM orders o
        JOIN shipping_addresses sa ON o.shipping_address_id = sa.address_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY sa.state, p.name
    )
    SELECT 
        sa.state,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(o.shipping_cost) as total_shipping_cost,
        AVG(o.shipping_cost) as avg_shipping_cost,
        SUM(oi.quantity * p.weight) / COUNT(DISTINCT o.order_id) as avg_order_weight,
        GROUP_CONCAT(
            CASE WHEN sp.rn <= 3 THEN sp.product_name END, 
            ', '
        ) as top_products
    FROM orders o
    JOIN shipping_addresses sa ON o.shipping_address_id = sa.address_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN state_products sp ON sa.state = sp.state AND p.name = sp.product_name
    GROUP BY sa.state
    ORDER BY total_shipping_cost DESC
    `;

  const rows: any[] = await db.all(query);
  return rows as ShippingCostByState[];
}

export async function findDeliveryDelays(
  db: Database,
  expectedDays: number = 5
): Promise<DeliveryDelay[]> {
  const query = `
    SELECT 
        o.order_id,
        o.created_at as order_date,
        o.status as current_status,
        CAST((julianday('now') - julianday(o.created_at)) AS INTEGER) as days_since_order,
        c.email,
        c.phone,
        c.segment as customer_segment,
        sa.city || ', ' || sa.state as destination,
        GROUP_CONCAT(p.name || ' (x' || oi.quantity || ')', ', ') as products,
        o.total_amount,
        w.name as last_known_warehouse
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN shipping_addresses sa ON o.shipping_address_id = sa.address_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN warehouses w ON o.warehouse_id = w.warehouse_id
    WHERE o.status NOT IN ('delivered', 'cancelled')
    AND julianday('now') - julianday(o.created_at) > ?
    GROUP BY o.order_id, o.created_at, o.status, c.email, c.phone, 
             c.segment, sa.city, sa.state, o.total_amount, w.name
    ORDER BY days_since_order DESC
    `;

  const rows: any[] = await db.all(query, [expectedDays]);
  return rows as DeliveryDelay[];
}
