import { open } from "sqlite";
import sqlite3 from "sqlite3";

import { createSchema } from "./schema";

async function main() {
  const db = await open({
    filename: "ecommerce.db",
    driver: sqlite3.Database,
  });

  await createSchema(db);
}

main();
