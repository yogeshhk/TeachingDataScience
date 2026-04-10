import { test, expect, vi } from "vitest";
import {
  transformJSX,
  createBlobURL,
  createImportMap,
  createPreviewHTML,
} from "../jsx-transformer";
import * as Babel from "@babel/standalone";

// Mock @babel/standalone
vi.mock("@babel/standalone", () => ({
  transform: vi.fn((code, options) => {
    // Simple mock that returns the code with some transformations
    if (options.filename?.endsWith(".tsx") || options.filename?.endsWith(".ts")) {
      return { code: code.replace(/const/g, "var") };
    }
    return { code };
  }),
}));

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn((blob) => {
  return `blob:mock-url-${Math.random()}`;
});

test("transformJSX transforms TypeScript files with correct presets", () => {
  const code = `const Component = () => <div>Hello</div>;`;
  const result = transformJSX(code, "test.tsx", new Set());

  expect(result.error).toBeUndefined();
  expect(result.code).toBe("var Component = () => <div>Hello</div>;");
  expect(result.missingImports).toBeDefined();
});

test("transformJSX handles JavaScript files without TypeScript preset", () => {
  const code = `const Component = () => <div>Hello</div>;`;
  const result = transformJSX(code, "test.jsx", new Set());

  expect(result.error).toBeUndefined();
  expect(result.code).toBe(code);
  expect(result.missingImports).toBeDefined();
});

test("transformJSX collects imports from code", () => {
  const code = `
    import React from 'react';
    import { useState } from 'react';
    import Component from './Component';
    import { utils } from '../utils';
  `;
  const result = transformJSX(code, "test.jsx", new Set());

  expect(result.missingImports).toContain("react");
  expect(result.missingImports).toContain("./Component");
  expect(result.missingImports).toContain("../utils");
  expect(result.missingImports?.size).toBe(3);
});

test("transformJSX handles transform errors gracefully", () => {
  // Mock Babel to throw an error
  vi.mocked(Babel.transform).mockImplementationOnce(() => {
    throw new Error("Transform failed");
  });

  const result = transformJSX("invalid code", "test.jsx", new Set());

  expect(result.code).toBe("");
  expect(result.error).toBe("Transform failed");
  
  // Reset the mock
  vi.mocked(Babel.transform).mockReset();
});

test("createBlobURL creates blob with correct mime type", () => {
  const code = "console.log('test');";
  const url = createBlobURL(code);

  expect(URL.createObjectURL).toHaveBeenCalledWith(
    expect.objectContaining({
      type: "application/javascript",
    })
  );
  expect(url).toMatch(/^blob:mock-url-/);
});

test("createBlobURL accepts custom mime type", () => {
  const code = "body { color: red; }";
  createBlobURL(code, "text/css");

  expect(URL.createObjectURL).toHaveBeenCalledWith(
    expect.objectContaining({
      type: "text/css",
    })
  );
});

test("createImportMap includes React CDN imports", () => {
  const files = new Map();
  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  expect(parsed.imports).toHaveProperty("react", "https://esm.sh/react@19");
  expect(parsed.imports).toHaveProperty("react-dom", "https://esm.sh/react-dom@19");
  expect(parsed.imports).toHaveProperty("react-dom/client", "https://esm.sh/react-dom@19/client");
  expect(parsed.imports).toHaveProperty("react/jsx-runtime", "https://esm.sh/react@19/jsx-runtime");
});

test("createImportMap transforms JavaScript and TypeScript files", () => {
  const files = new Map([
    ["/App.jsx", "export default function App() { return <div>App</div>; }"],
    ["/utils.ts", "export const helper = () => {};"],
    ["/styles.css", "body { margin: 0; }"],
  ]);

  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  // Should have blob URLs for JS/TS files
  expect(parsed.imports["/App.jsx"]).toMatch(/^blob:mock-url-/);
  expect(parsed.imports["/utils.ts"]).toMatch(/^blob:mock-url-/);
  
  // Should not have CSS files
  expect(parsed.imports["/styles.css"]).toBeUndefined();
});

test("createImportMap creates multiple path variations for files", () => {
  const files = new Map([
    ["/components/Button.jsx", "export default function Button() {}"],
  ]);

  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  // All these variations should point to the same blob URL
  const blobUrl = parsed.imports["/components/Button.jsx"];
  expect(parsed.imports["components/Button.jsx"]).toBe(blobUrl);
  expect(parsed.imports["@/components/Button.jsx"]).toBe(blobUrl);
  expect(parsed.imports["@/components/Button.jsx"]).toBe(blobUrl);
  expect(parsed.imports["/components/Button"]).toBe(blobUrl);
  expect(parsed.imports["components/Button"]).toBe(blobUrl);
  expect(parsed.imports["@/components/Button"]).toBe(blobUrl);
});

test("createImportMap creates placeholder modules for missing imports", () => {
  const files = new Map([
    ["/App.jsx", "import Button from './components/Button'; export default function App() {}"],
  ]);

  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  // Should create placeholder for missing Button component
  expect(parsed.imports["./components/Button"]).toBeDefined();
  expect(parsed.imports["./components/Button"]).toMatch(/^blob:mock-url-/);
});

test("createImportMap handles @/ alias imports", () => {
  const files = new Map([
    ["/App.jsx", "import { utils } from '@/lib/utils'; export default function App() {}"],
  ]);

  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  // Should create placeholder with proper variations
  expect(parsed.imports["@/lib/utils"]).toBeDefined();
  expect(parsed.imports["/lib/utils"]).toBeDefined();
  expect(parsed.imports["lib/utils"]).toBeDefined();
});

test("createPreviewHTML generates valid HTML with import map", () => {
  const importMap = JSON.stringify({
    imports: {
      "/App.jsx": "blob:mock-url-123",
      "react": "https://esm.sh/react@19",
    },
  });

  const html = createPreviewHTML("/App.jsx", importMap);

  expect(html).toContain("<!DOCTYPE html>");
  expect(html).toContain('<div id="root"></div>');
  expect(html).toContain('type="importmap"');
  expect(html).toContain(importMap);
  expect(html).toContain("blob:mock-url-123");
  expect(html).toContain("import('blob:mock-url-123')");
});

test("createPreviewHTML includes Tailwind CSS", () => {
  const html = createPreviewHTML("/App.jsx", "{}");
  expect(html).toContain("https://cdn.tailwindcss.com");
});

test("createPreviewHTML includes error boundary", () => {
  const html = createPreviewHTML("/App.jsx", "{}");
  expect(html).toContain("class ErrorBoundary");
  expect(html).toContain("componentDidCatch");
  expect(html).toContain("error-boundary");
});

test("createPreviewHTML handles invalid import map gracefully", () => {
  // Mock console.error to prevent noise in test output
  const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  
  const invalidImportMap = "{ invalid json";
  const html = createPreviewHTML("/App.jsx", invalidImportMap);

  // Should still generate HTML with the entry point
  expect(html).toContain("/App.jsx");
  // The error is logged to console, not included in HTML
  expect(html).toContain("console.error('Failed to load app:', error)");
  
  // Restore console.error
  consoleErrorSpy.mockRestore();
});

test("integration: full transformation pipeline works", () => {
  const files = new Map([
    ["/App.jsx", `
      import React from 'react';
      import Button from './Button';
      
      export default function App() {
        return <div><Button /></div>;
      }
    `],
    ["/Button.jsx", `
      export default function Button() {
        return <button>Click me</button>;
      }
    `],
  ]);

  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);

  // Both components should be transformed
  expect(parsed.imports["/App.jsx"]).toMatch(/^blob:mock-url-/);
  expect(parsed.imports["/Button.jsx"]).toMatch(/^blob:mock-url-/);

  // Import variations should exist
  expect(parsed.imports["./Button"]).toBeDefined();
  expect(parsed.imports["/Button"]).toBeDefined();

  // Create preview HTML
  const html = createPreviewHTML("/App.jsx", result.importMap);
  expect(html).toContain(parsed.imports["/App.jsx"]);
});

// CSS Support Tests
test("transformJSX detects CSS imports", () => {
  const code = `
    import React from 'react';
    import './styles.css';
    import '@/styles/globals.css';
    import "../components/Button.css";
    
    export default function App() { return <div>App</div>; }
  `;
  const result = transformJSX(code, "App.jsx", new Set());
  
  expect(result.cssImports).toBeDefined();
  expect(result.cssImports).toContain("./styles.css");
  expect(result.cssImports).toContain("@/styles/globals.css");
  expect(result.cssImports).toContain("../components/Button.css");
});

test("transformJSX removes CSS imports from transformed code", () => {
  const code = `
    import React from 'react';
    import './styles.css';
    
    export default function App() { return <div>App</div>; }
  `;
  const result = transformJSX(code, "App.jsx", new Set());
  
  expect(result.code).not.toContain("import './styles.css'");
  expect(result.code).toContain("React");
});

test("transformJSX handles CSS imports with different quotes", () => {
  const code = `
    import './single.css';
    import "./double.css";
    import '@/styles/globals.css';
  `;
  const result = transformJSX(code, "App.jsx", new Set());
  
  expect(result.cssImports).toContain("./single.css");
  expect(result.cssImports).toContain("./double.css");
  expect(result.cssImports).toContain("@/styles/globals.css");
});

test("createImportMap collects CSS files and returns styles", () => {
  const files = new Map([
    ["/App.jsx", `import './styles.css'; export default function App() {}`],
    ["/styles.css", `body { margin: 0; } .container { padding: 20px; }`],
    ["/globals.css", `* { box-sizing: border-box; }`],
  ]);
  
  const result = createImportMap(files);
  
  // Should return an object with imports and styles
  expect(result).toHaveProperty("importMap");
  expect(result).toHaveProperty("styles");
  
  // Should collect CSS content
  expect(result.styles).toContain("body { margin: 0; }");
  expect(result.styles).toContain("* { box-sizing: border-box; }");
});

test("createImportMap handles missing CSS files gracefully", () => {
  const files = new Map([
    ["/App.jsx", `import './missing.css'; export default function App() {}`],
  ]);
  
  const result = createImportMap(files);
  
  // Should not throw error
  expect(result.styles).toBeDefined();
  // Could include comment about missing file
  expect(result.styles).toContain("/* ./missing.css not found */");
});

test("createImportMap resolves CSS import paths correctly", () => {
  const files = new Map([
    ["/src/App.jsx", `import '@/styles/globals.css'; export default function App() {}`],
    ["/styles/globals.css", `body { background: white; }`],
  ]);
  
  const result = createImportMap(files);
  expect(result.styles).toContain("body { background: white; }");
});

test("createPreviewHTML injects CSS styles into head", () => {
  const styles = `
    body { margin: 0; }
    .container { padding: 20px; }
  `;
  
  const html = createPreviewHTML("/App.jsx", "{}", styles);
  
  expect(html).toContain("<style>");
  expect(html).toContain("body { margin: 0; }");
  expect(html).toContain(".container { padding: 20px; }");
});

test("createPreviewHTML handles empty CSS gracefully", () => {
  const html = createPreviewHTML("/App.jsx", "{}", "");
  
  // Should not break without styles
  expect(html).toContain("<!DOCTYPE html>");
  expect(html).toContain('<div id="root"></div>');
});

test("createPreviewHTML preserves existing styles with CSS injection", () => {
  const customStyles = "h1 { color: blue; }";
  const html = createPreviewHTML("/App.jsx", "{}", customStyles);
  
  // Should have both Tailwind and custom styles
  expect(html).toContain("https://cdn.tailwindcss.com");
  expect(html).toContain("h1 { color: blue; }");
  // Existing styles should remain
  expect(html).toContain("body {");
  expect(html).toContain(".error-boundary {");
});

test("integration: full pipeline handles components with CSS imports", () => {
  const files = new Map([
    ["/App.jsx", `
      import React from 'react';
      import './App.css';
      import '@/styles/globals.css';
      
      export default function App() {
        return <div className="container">Hello</div>;
      }
    `],
    ["/App.css", `.container { max-width: 1200px; margin: 0 auto; }`],
    ["/styles/globals.css", `body { font-family: sans-serif; }`],
  ]);
  
  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);
  
  // JS files should be in import map
  expect(parsed.imports["/App.jsx"]).toMatch(/^blob:mock-url-/);
  
  // CSS should be collected
  expect(result.styles).toContain(".container { max-width: 1200px;");
  expect(result.styles).toContain("body { font-family: sans-serif;");
  
  // HTML should include CSS
  const html = createPreviewHTML("/App.jsx", result.importMap, result.styles);
  expect(html).toContain(".container { max-width: 1200px;");
});

// Error handling tests
test("createImportMap handles syntax errors gracefully", () => {
  // Mock Babel to throw error for BadComponent
  vi.mocked(Babel.transform).mockImplementation((code, options) => {
    if (options.filename === "/BadComponent.jsx") {
      throw new Error("Unexpected token: Missing closing tag");
    }
    // Return transformed code for other files
    if (options.filename?.endsWith(".tsx") || options.filename?.endsWith(".ts")) {
      return { code: code.replace(/const/g, "var") };
    }
    return { code };
  });
  
  const files = new Map([
    ["/App.jsx", `export default function App() { return <div>Hello</div>; }`],
    ["/BadComponent.jsx", `
      export default function BadComponent() {
        return <div>Missing closing tag
      }
    `],
  ]);
  
  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);
  
  // Good file should be in import map
  expect(parsed.imports["/App.jsx"]).toMatch(/^blob:mock-url-/);
  // Bad file should NOT be in import map anymore
  expect(parsed.imports["/BadComponent.jsx"]).toBeUndefined();
  
  // Should have error for BadComponent
  expect(result.errors).toHaveLength(1);
  expect(result.errors[0].path).toBe("/BadComponent.jsx");
  expect(result.errors[0].error).toBe("Unexpected token: Missing closing tag");
  
  // Restore mock
  vi.mocked(Babel.transform).mockReset();
});

test("createPreviewHTML displays syntax errors", () => {
  const errors = [
    { path: "/Component.jsx", error: "Unexpected token" },
    { path: "/Another.jsx", error: "Missing semicolon" }
  ];
  
  const html = createPreviewHTML("/App.jsx", "{}", "", errors);
  
  // Should show error section
  expect(html).toContain("Syntax Errors (2)");
  expect(html).toContain("/Component.jsx");
  expect(html).toContain("Unexpected token");
  expect(html).toContain("/Another.jsx");
  expect(html).toContain("Missing semicolon");
  
  // Should NOT include the app script when there are errors
  expect(html).not.toContain("loadApp()");
});

test("files with syntax errors are not included in import map", () => {
  // Mock Babel to throw error for BadComponent
  vi.mocked(Babel.transform).mockImplementation((code, options) => {
    if (options.filename === "/BadComponent.jsx") {
      throw new Error("Syntax error in BadComponent");
    }
    // Return default mock behavior for other files
    return { code };
  });
  
  const files = new Map([
    ["/App.jsx", `
      import BadComponent from './BadComponent';
      export default function App() { 
        return <div><BadComponent /></div>; 
      }
    `],
    ["/BadComponent.jsx", `
      export default function BadComponent() {
        return <div>Missing closing tag
      }
    `],
  ]);
  
  const result = createImportMap(files);
  const parsed = JSON.parse(result.importMap);
  
  // BadComponent should NOT be in import map anymore
  expect(parsed.imports["/BadComponent.jsx"]).toBeUndefined();
  expect(parsed.imports["/BadComponent"]).toBeUndefined();
  
  // But a placeholder should be created for the import
  expect(parsed.imports["./BadComponent"]).toBeDefined();
  
  // Should have error tracked
  expect(result.errors.some(e => e.path === "/BadComponent.jsx")).toBe(true);
  
  // Restore mock
  vi.mocked(Babel.transform).mockReset();
});