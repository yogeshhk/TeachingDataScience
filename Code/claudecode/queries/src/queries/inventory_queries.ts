import { Database } from "sqlite";

interface WarehouseInventoryItem {
  product_id: number;
  product_name: string;
  sku: string;
  category_name: string;
  quantity: number;
  reserved_quantity: number;
  available_quantity: number;
  last_restocked_at: string;
  days_since_restock: number;
}

interface ProductAvailability {
  warehouse_id: number;
  warehouse_name: string;
  warehouse_city: string;
  warehouse_state: string;
  product_name: string;
  sku: string;
  product_price: number;
  total_quantity: number;
  reserved_quantity: number;
  available_quantity: number;
  last_restocked_at: string;
}

interface StockTransfer {
  product_id: number;
  product_name: string;
  sku: string;
  category_name: string;
  low_stock_warehouse_id: number;
  low_stock_warehouse_name: string;
  low_stock_quantity: number;
  low_stock_sales_14d: number;
  high_stock_warehouse_id: number;
  high_stock_warehouse_name: string;
  high_stock_quantity: number;
  high_stock_sales_14d: number;
}

interface WarehouseInventoryValue {
  warehouse_id: number;
  warehouse_name: string;
  city: string;
  state: string;
  country: string;
  unique_skus: number;
  total_inventory_value: number;
  total_units: number;
  total_reserved_units: number;
  oldest_stock_date: string;
  newest_stock_date: string;
}

interface ReservedInventoryItem {
  product_id: number;
  product_name: string;
  sku: string;
  product_cost: number;
  product_price: number;
  warehouse_id: number;
  warehouse_name: string;
  reserved_quantity: number;
  reserved_cost: number;
  pending_order_count: number;
  customer_segments: string;
}

interface InventoryMovement {
  product_id: number;
  product_name: string;
  sku: string;
  category_name: string;
  warehouse_id: number;
  warehouse_name: string;
  current_quantity: number;
  last_restocked_at: string;
  last_restock_quantity: number;
  supplier: string;
  units_sold_period: number;
  turnover_rate: number;
}

export async function getWarehouseInventory(
  db: Database,
  warehouseId: number
): Promise<WarehouseInventoryItem[]> {
  const query = `
    SELECT 
        i.product_id,
        p.name AS product_name,
        p.sku,
        c.name AS category_name,
        i.quantity,
        i.reserved_quantity,
        i.quantity - i.reserved_quantity AS available_quantity,
        i.last_restocked_at,
        CAST((julianday('now') - julianday(i.last_restocked_at)) AS INTEGER) AS days_since_restock
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    WHERE i.warehouse_id = ?
    ORDER BY i.quantity DESC
    `;

  const rows: any[] = await db.all(query, [warehouseId]);
  return rows as WarehouseInventoryItem[];
}

export async function checkProductAvailability(
  db: Database,
  productId: number
): Promise<ProductAvailability[]> {
  const query = `
    SELECT 
        i.warehouse_id,
        w.name AS warehouse_name,
        w.city AS warehouse_city,
        w.state AS warehouse_state,
        p.name AS product_name,
        p.sku,
        p.price AS product_price,
        i.quantity AS total_quantity,
        i.reserved_quantity,
        i.quantity - i.reserved_quantity AS available_quantity,
        i.last_restocked_at
    FROM inventory i
    JOIN warehouses w ON i.warehouse_id = w.warehouse_id
    JOIN products p ON i.product_id = p.product_id
    WHERE i.product_id = ?
    ORDER BY available_quantity DESC
    `;

  const rows: any[] = await db.all(query, [productId]);
  return rows as ProductAvailability[];
}

export async function findStockTransfersNeeded(
  db: Database
): Promise<StockTransfer[]> {
  const query = `
    WITH warehouse_sales AS (
        SELECT 
            oi.product_id,
            o.warehouse_id,
            COUNT(DISTINCT o.order_id) AS orders_14d,
            SUM(oi.quantity) AS units_sold_14d
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.created_at >= datetime('now', '-14 days')
        GROUP BY oi.product_id, o.warehouse_id
    ),
    inventory_status AS (
        SELECT 
            i.product_id,
            i.warehouse_id,
            p.name AS product_name,
            p.sku,
            c.name AS category_name,
            i.quantity - i.reserved_quantity AS available_quantity,
            COALESCE(ws.units_sold_14d, 0) AS units_sold_14d
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        LEFT JOIN warehouse_sales ws 
            ON i.product_id = ws.product_id 
            AND i.warehouse_id = ws.warehouse_id
    )
    SELECT 
        low.product_id,
        low.product_name,
        low.sku,
        low.category_name,
        low.warehouse_id AS low_stock_warehouse_id,
        w1.name AS low_stock_warehouse_name,
        low.available_quantity AS low_stock_quantity,
        low.units_sold_14d AS low_stock_sales_14d,
        high.warehouse_id AS high_stock_warehouse_id,
        w2.name AS high_stock_warehouse_name,
        high.available_quantity AS high_stock_quantity,
        high.units_sold_14d AS high_stock_sales_14d
    FROM inventory_status low
    JOIN inventory_status high 
        ON low.product_id = high.product_id 
        AND low.warehouse_id != high.warehouse_id
    JOIN warehouses w1 ON low.warehouse_id = w1.warehouse_id
    JOIN warehouses w2 ON high.warehouse_id = w2.warehouse_id
    WHERE low.available_quantity < 20
        AND high.available_quantity > 50
        AND low.units_sold_14d > 0
    ORDER BY low.units_sold_14d DESC, high.available_quantity DESC
    `;

  const rows: any[] = await db.all(query, []);
  return rows as StockTransfer[];
}

export async function getInventoryValueByWarehouse(
  db: Database
): Promise<WarehouseInventoryValue[]> {
  const query = `
    SELECT 
        w.warehouse_id,
        w.name AS warehouse_name,
        w.city,
        w.state,
        w.country,
        COUNT(DISTINCT i.product_id) AS unique_skus,
        SUM(i.quantity * p.price) AS total_inventory_value,
        SUM(i.quantity) AS total_units,
        SUM(i.reserved_quantity) AS total_reserved_units,
        MIN(i.last_restocked_at) AS oldest_stock_date,
        MAX(i.last_restocked_at) AS newest_stock_date
    FROM warehouses w
    LEFT JOIN inventory i ON w.warehouse_id = i.warehouse_id
    LEFT JOIN products p ON i.product_id = p.product_id
    GROUP BY w.warehouse_id, w.name, w.city, w.state, w.country
    ORDER BY total_inventory_value DESC
    `;

  const rows: any[] = await db.all(query, []);
  return rows as WarehouseInventoryValue[];
}

export async function fetchReservedInventory(
  db: Database
): Promise<ReservedInventoryItem[]> {
  const query = `
    WITH reserved_orders AS (
        SELECT 
            oi.product_id,
            o.warehouse_id,
            COUNT(DISTINCT o.order_id) AS pending_order_count,
            COUNT(DISTINCT c.segment) AS unique_segments,
            GROUP_CONCAT(DISTINCT c.segment) AS customer_segments
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.status IN ('pending', 'processing')
        GROUP BY oi.product_id, o.warehouse_id
    )
    SELECT 
        i.product_id,
        p.name AS product_name,
        p.sku,
        p.cost AS product_cost,
        p.price AS product_price,
        i.warehouse_id,
        w.name AS warehouse_name,
        i.reserved_quantity,
        i.reserved_quantity * p.cost AS reserved_cost,
        ro.pending_order_count,
        ro.customer_segments
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    JOIN warehouses w ON i.warehouse_id = w.warehouse_id
    LEFT JOIN reserved_orders ro 
        ON i.product_id = ro.product_id 
        AND i.warehouse_id = ro.warehouse_id
    WHERE i.reserved_quantity > 0
    ORDER BY i.reserved_quantity DESC
    `;

  const rows: any[] = await db.all(query, []);
  return rows as ReservedInventoryItem[];
}

export async function getInventoryMovements(
  db: Database,
  days: number = 30
): Promise<InventoryMovement[]> {
  const query = `
    WITH recent_sales AS (
        SELECT 
            oi.product_id,
            o.warehouse_id,
            SUM(oi.quantity) AS units_sold
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.created_at >= datetime('now', ? || ' days')
        GROUP BY oi.product_id, o.warehouse_id
    ),
    turnover_data AS (
        SELECT 
            i.product_id,
            i.warehouse_id,
            i.quantity AS current_stock,
            COALESCE(rs.units_sold, 0) AS units_sold,
            CASE 
                WHEN i.quantity > 0 THEN 
                    ROUND(CAST(COALESCE(rs.units_sold, 0) AS FLOAT) / i.quantity, 2)
                ELSE 0 
            END AS turnover_rate
        FROM inventory i
        LEFT JOIN recent_sales rs 
            ON i.product_id = rs.product_id 
            AND i.warehouse_id = rs.warehouse_id
    )
    SELECT 
        i.product_id,
        p.name AS product_name,
        p.sku,
        c.name AS category_name,
        i.warehouse_id,
        w.name AS warehouse_name,
        i.quantity AS current_quantity,
        i.last_restocked_at,
        i.restock_quantity AS last_restock_quantity,
        CASE 
            WHEN i.description LIKE '%from %' THEN 
                SUBSTR(i.description, INSTR(i.description, 'from ') + 5)
            ELSE 'Unknown Supplier'
        END AS supplier,
        td.units_sold AS units_sold_period,
        td.turnover_rate
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    JOIN warehouses w ON i.warehouse_id = w.warehouse_id
    LEFT JOIN turnover_data td 
        ON i.product_id = td.product_id 
        AND i.warehouse_id = td.warehouse_id
    WHERE i.last_restocked_at >= datetime('now', ? || ' days')
    ORDER BY i.last_restocked_at DESC
    `;

  const rows: any[] = await db.all(query, [-days, -days]);
  return rows as InventoryMovement[];
}
