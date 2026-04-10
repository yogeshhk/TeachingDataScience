import { test, expect } from "vitest";
import { VirtualFileSystem } from "@/lib/file-system";

test("creates a new file system with root directory", () => {
  const fs = new VirtualFileSystem();
  const root = fs.getNode("/");

  expect(root).toBeDefined();
  expect(root?.type).toBe("directory");
  expect(root?.name).toBe("/");
  expect(root?.path).toBe("/");
});

test("normalizes paths correctly", () => {
  const fs = new VirtualFileSystem();

  // Test by creating files with different path formats
  fs.createFile("test.txt", "content");
  expect(fs.exists("/test.txt")).toBe(true);

  // Test path normalization - createFile now creates parent directories
  fs.createFile("//folder//file.txt", "content");
  expect(fs.exists("/folder/file.txt")).toBe(true);

  fs.createFile("/trailing/", "content");
  expect(fs.exists("/trailing")).toBe(true);
});

test("creates files in root directory", () => {
  const fs = new VirtualFileSystem();
  const file = fs.createFile("/test.txt", "Hello World");

  expect(file).toBeDefined();
  expect(file?.type).toBe("file");
  expect(file?.name).toBe("test.txt");
  expect(file?.path).toBe("/test.txt");
  expect(file?.content).toBe("Hello World");
});

test("creates directories", () => {
  const fs = new VirtualFileSystem();
  const dir = fs.createDirectory("/src");

  expect(dir).toBeDefined();
  expect(dir?.type).toBe("directory");
  expect(dir?.name).toBe("src");
  expect(dir?.path).toBe("/src");
  expect(dir?.children).toBeDefined();
});

test("creates nested files and directories", () => {
  const fs = new VirtualFileSystem();

  // createFile now automatically creates parent directories
  const file = fs.createFile(
    "/src/components/Button.tsx",
    "export const Button = () => {};"
  );

  expect(file).toBeDefined();
  expect(file?.path).toBe("/src/components/Button.tsx");
  expect(fs.exists("/src")).toBe(true);
  expect(fs.exists("/src/components")).toBe(true);
  expect(fs.readFile("/src/components/Button.tsx")).toBe(
    "export const Button = () => {};"
  );
});

test("returns null when creating existing file", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "content");
  const duplicate = fs.createFile("/test.txt", "new content");

  expect(duplicate).toBeNull();
  expect(fs.readFile("/test.txt")).toBe("content");
});

test("returns null when creating existing directory", () => {
  const fs = new VirtualFileSystem();

  fs.createDirectory("/src");
  const duplicate = fs.createDirectory("/src");

  expect(duplicate).toBeNull();
});

test("creates file with parent directories automatically", () => {
  const fs = new VirtualFileSystem();
  const file = fs.createFile("/nonexistent/test.txt", "content");

  expect(file).toBeDefined();
  expect(file?.path).toBe("/nonexistent/test.txt");
  expect(fs.exists("/nonexistent")).toBe(true);
  expect(fs.readFile("/nonexistent/test.txt")).toBe("content");
});

test("creates deeply nested file with all parent directories", () => {
  const fs = new VirtualFileSystem();
  const file = fs.createFile("/a/b/c/d/e/file.txt", "deep content");

  expect(file).toBeDefined();
  expect(file?.path).toBe("/a/b/c/d/e/file.txt");
  expect(fs.exists("/a")).toBe(true);
  expect(fs.exists("/a/b")).toBe(true);
  expect(fs.exists("/a/b/c")).toBe(true);
  expect(fs.exists("/a/b/c/d")).toBe(true);
  expect(fs.exists("/a/b/c/d/e")).toBe(true);
  expect(fs.readFile("/a/b/c/d/e/file.txt")).toBe("deep content");
});

test("reads file content", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "Hello World");
  expect(fs.readFile("/test.txt")).toBe("Hello World");

  fs.createFile("/empty.txt", "");
  expect(fs.readFile("/empty.txt")).toBe("");
});

test("returns null when reading non-existent file", () => {
  const fs = new VirtualFileSystem();
  expect(fs.readFile("/nonexistent.txt")).toBeNull();
});

test("returns null when reading directory as file", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");
  expect(fs.readFile("/src")).toBeNull();
});

test("updates file content", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "original");
  const updated = fs.updateFile("/test.txt", "updated");

  expect(updated).toBe(true);
  expect(fs.readFile("/test.txt")).toBe("updated");
});

test("returns false when updating non-existent file", () => {
  const fs = new VirtualFileSystem();
  const updated = fs.updateFile("/nonexistent.txt", "content");

  expect(updated).toBe(false);
});

test("returns false when updating directory", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");
  const updated = fs.updateFile("/src", "content");

  expect(updated).toBe(false);
});

test("deletes files", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "content");
  const deleted = fs.deleteFile("/test.txt");

  expect(deleted).toBe(true);
  expect(fs.exists("/test.txt")).toBe(false);
});

test("deletes directories recursively", () => {
  const fs = new VirtualFileSystem();

  fs.createDirectory("/src");
  fs.createDirectory("/src/components");
  fs.createFile("/src/components/Button.tsx", "content");
  fs.createFile("/src/index.ts", "content");

  const deleted = fs.deleteFile("/src");

  expect(deleted).toBe(true);
  expect(fs.exists("/src")).toBe(false);
  expect(fs.exists("/src/components")).toBe(false);
  expect(fs.exists("/src/components/Button.tsx")).toBe(false);
  expect(fs.exists("/src/index.ts")).toBe(false);
});

test("returns false when deleting non-existent file", () => {
  const fs = new VirtualFileSystem();
  expect(fs.deleteFile("/nonexistent.txt")).toBe(false);
});

test("returns false when deleting root directory", () => {
  const fs = new VirtualFileSystem();
  expect(fs.deleteFile("/")).toBe(false);
});

test("checks if path exists", () => {
  const fs = new VirtualFileSystem();

  expect(fs.exists("/")).toBe(true);
  expect(fs.exists("/nonexistent")).toBe(false);

  fs.createFile("/test.txt", "content");
  expect(fs.exists("/test.txt")).toBe(true);

  fs.createDirectory("/src");
  expect(fs.exists("/src")).toBe(true);
});

test("lists directory contents", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "content");
  fs.createDirectory("/src");
  fs.createFile("/README.md", "content");

  const contents = fs.listDirectory("/");

  expect(contents).toHaveLength(3);
  expect(contents?.map((n) => n.name).sort()).toEqual([
    "README.md",
    "src",
    "test.txt",
  ]);
});

test("returns null when listing non-directory", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  expect(fs.listDirectory("/test.txt")).toBeNull();
  expect(fs.listDirectory("/nonexistent")).toBeNull();
});

test("returns empty array for empty directory", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/empty");

  const contents = fs.listDirectory("/empty");
  expect(contents).toEqual([]);
});

test("gets all files as a map", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "content1");
  fs.createDirectory("/src");
  fs.createFile("/src/index.ts", "content2");
  fs.createFile("/src/app.ts", "content3");

  const files = fs.getAllFiles();

  expect(files.size).toBe(3);
  expect(files.get("/test.txt")).toBe("content1");
  expect(files.get("/src/index.ts")).toBe("content2");
  expect(files.get("/src/app.ts")).toBe("content3");
  expect(files.has("/src")).toBe(false);
});

test("serializes file system to plain object", () => {
  const fs = new VirtualFileSystem();

  fs.createFile("/test.txt", "content");
  fs.createDirectory("/src");
  fs.createFile("/src/index.ts", "export {}");

  const serialized = fs.serialize();

  expect(serialized["/"].type).toBe("directory");
  expect(serialized["/test.txt"].type).toBe("file");
  expect(serialized["/test.txt"].content).toBe("content");
  expect(serialized["/src"].type).toBe("directory");
  expect(serialized["/src/index.ts"].type).toBe("file");
  expect(serialized["/src/index.ts"].content).toBe("export {}");
});

test("deserializes from file map", () => {
  const fs = new VirtualFileSystem();

  const data = {
    "/test.txt": "content1",
    "/src/components/Button.tsx": "button content",
    "/src/index.ts": "index content",
  };

  fs.deserialize(data);

  expect(fs.readFile("/test.txt")).toBe("content1");
  expect(fs.readFile("/src/components/Button.tsx")).toBe("button content");
  expect(fs.readFile("/src/index.ts")).toBe("index content");
  expect(fs.exists("/src")).toBe(true);
  expect(fs.exists("/src/components")).toBe(true);
});

test("deserializes from node map", () => {
  const fs = new VirtualFileSystem();

  const data = {
    "/": { type: "directory" as const, name: "/", path: "/" },
    "/src": { type: "directory" as const, name: "src", path: "/src" },
    "/src/index.ts": {
      type: "file" as const,
      name: "index.ts",
      path: "/src/index.ts",
      content: "export {}",
    },
  };

  fs.deserializeFromNodes(data);

  expect(fs.exists("/src")).toBe(true);
  expect(fs.readFile("/src/index.ts")).toBe("export {}");
});

test("viewFile shows file content with line numbers", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2\nline3");

  const view = fs.viewFile("/test.txt");
  expect(view).toBe("1\tline1\n2\tline2\n3\tline3");
});

test("viewFile with view range", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2\nline3\nline4\nline5");

  const view = fs.viewFile("/test.txt", [2, 4]);
  expect(view).toBe("2\tline2\n3\tline3\n4\tline4");

  const viewToEnd = fs.viewFile("/test.txt", [3, -1]);
  expect(viewToEnd).toBe("3\tline3\n4\tline4\n5\tline5");
});

test("viewFile shows directory contents", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");
  fs.createFile("/src/index.ts", "");
  fs.createDirectory("/src/components");

  const view = fs.viewFile("/src");
  expect(view).toBe("[DIR] components\n[FILE] index.ts");
});

test("viewFile shows empty directory", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/empty");

  const view = fs.viewFile("/empty");
  expect(view).toBe("(empty directory)");
});

test("viewFile shows empty file", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/empty.txt", "");

  const view = fs.viewFile("/empty.txt");
  expect(view).toBe("1\t");
});

test("viewFile returns error for non-existent path", () => {
  const fs = new VirtualFileSystem();

  const view = fs.viewFile("/nonexistent");
  expect(view).toBe("File not found: /nonexistent");
});

test("createFileWithParents creates parent directories", () => {
  const fs = new VirtualFileSystem();

  const result = fs.createFileWithParents(
    "/src/components/Button.tsx",
    "content"
  );

  expect(result).toBe("File created: /src/components/Button.tsx");
  expect(fs.exists("/src")).toBe(true);
  expect(fs.exists("/src/components")).toBe(true);
  expect(fs.readFile("/src/components/Button.tsx")).toBe("content");
});

test("createFileWithParents returns error for existing file", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  const result = fs.createFileWithParents("/test.txt", "new content");
  expect(result).toBe("Error: File already exists: /test.txt");
});

test("replaceInFile replaces all occurrences", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "foo bar foo baz foo");

  const result = fs.replaceInFile("/test.txt", "foo", "hello");

  expect(result).toBe("Replaced 3 occurrence(s) of the string in /test.txt");
  expect(fs.readFile("/test.txt")).toBe("hello bar hello baz hello");
});

test("replaceInFile handles empty replacement", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "foo bar foo");

  const result = fs.replaceInFile("/test.txt", "foo", "");

  expect(result).toBe("Replaced 2 occurrence(s) of the string in /test.txt");
  expect(fs.readFile("/test.txt")).toBe(" bar ");
});

test("replaceInFile returns error for non-existent file", () => {
  const fs = new VirtualFileSystem();

  const result = fs.replaceInFile("/nonexistent.txt", "foo", "bar");
  expect(result).toBe("Error: File not found: /nonexistent.txt");
});

test("replaceInFile returns error for directory", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");

  const result = fs.replaceInFile("/src", "foo", "bar");
  expect(result).toBe("Error: Cannot edit a directory: /src");
});

test("replaceInFile returns error when string not found", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "hello world");

  const result = fs.replaceInFile("/test.txt", "foo", "bar");
  expect(result).toBe('Error: String not found in file: "foo"');
});

test("insertInFile inserts text at specified line", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2\nline3");

  const result = fs.insertInFile("/test.txt", 1, "inserted");

  expect(result).toBe("Text inserted at line 1 in /test.txt");
  expect(fs.readFile("/test.txt")).toBe("line1\ninserted\nline2\nline3");
});

test("insertInFile inserts at beginning", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2");

  const result = fs.insertInFile("/test.txt", 0, "first");

  expect(result).toBe("Text inserted at line 0 in /test.txt");
  expect(fs.readFile("/test.txt")).toBe("first\nline1\nline2");
});

test("insertInFile inserts at end", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2");

  const result = fs.insertInFile("/test.txt", 2, "last");

  expect(result).toBe("Text inserted at line 2 in /test.txt");
  expect(fs.readFile("/test.txt")).toBe("line1\nline2\nlast");
});

test("insertInFile returns error for invalid line number", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "line1\nline2");

  const result = fs.insertInFile("/test.txt", 5, "text");
  expect(result).toBe("Error: Invalid line number: 5. File has 2 lines.");

  const negativeResult = fs.insertInFile("/test.txt", -1, "text");
  expect(negativeResult).toBe(
    "Error: Invalid line number: -1. File has 2 lines."
  );
});

test("insertInFile returns error for non-existent file", () => {
  const fs = new VirtualFileSystem();

  const result = fs.insertInFile("/nonexistent.txt", 0, "text");
  expect(result).toBe("Error: File not found: /nonexistent.txt");
});

test("insertInFile returns error for directory", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");

  const result = fs.insertInFile("/src", 0, "text");
  expect(result).toBe("Error: Cannot edit a directory: /src");
});

test("rename moves a file to a new location", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  const result = fs.rename("/test.txt", "/renamed.txt");

  expect(result).toBe(true);
  expect(fs.exists("/test.txt")).toBe(false);
  expect(fs.exists("/renamed.txt")).toBe(true);
  expect(fs.readFile("/renamed.txt")).toBe("content");
});

test("rename moves a file to a different directory", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");
  fs.createDirectory("/docs");

  const result = fs.rename("/test.txt", "/docs/test.txt");

  expect(result).toBe(true);
  expect(fs.exists("/test.txt")).toBe(false);
  expect(fs.exists("/docs/test.txt")).toBe(true);
  expect(fs.readFile("/docs/test.txt")).toBe("content");
});

test("rename creates parent directories if they don't exist", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  const result = fs.rename("/test.txt", "/new/path/to/file.txt");

  expect(result).toBe(true);
  expect(fs.exists("/test.txt")).toBe(false);
  expect(fs.exists("/new")).toBe(true);
  expect(fs.exists("/new/path")).toBe(true);
  expect(fs.exists("/new/path/to")).toBe(true);
  expect(fs.exists("/new/path/to/file.txt")).toBe(true);
  expect(fs.readFile("/new/path/to/file.txt")).toBe("content");
});

test("rename moves a directory and all its contents", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");
  fs.createFile("/src/index.ts", "index content");
  fs.createDirectory("/src/components");
  fs.createFile("/src/components/Button.tsx", "button content");

  const result = fs.rename("/src", "/app");

  expect(result).toBe(true);
  expect(fs.exists("/src")).toBe(false);
  expect(fs.exists("/app")).toBe(true);
  expect(fs.exists("/app/index.ts")).toBe(true);
  expect(fs.exists("/app/components")).toBe(true);
  expect(fs.exists("/app/components/Button.tsx")).toBe(true);
  expect(fs.readFile("/app/index.ts")).toBe("index content");
  expect(fs.readFile("/app/components/Button.tsx")).toBe("button content");
});

test("rename moves a directory to a nested location with parent creation", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/src");
  fs.createFile("/src/file.txt", "content");

  const result = fs.rename("/src", "/deeply/nested/src");

  expect(result).toBe(true);
  expect(fs.exists("/src")).toBe(false);
  expect(fs.exists("/deeply")).toBe(true);
  expect(fs.exists("/deeply/nested")).toBe(true);
  expect(fs.exists("/deeply/nested/src")).toBe(true);
  expect(fs.exists("/deeply/nested/src/file.txt")).toBe(true);
  expect(fs.readFile("/deeply/nested/src/file.txt")).toBe("content");
});

test("rename returns false when source doesn't exist", () => {
  const fs = new VirtualFileSystem();

  const result = fs.rename("/nonexistent.txt", "/new.txt");

  expect(result).toBe(false);
  expect(fs.exists("/new.txt")).toBe(false);
});

test("rename returns false when destination already exists", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/source.txt", "source content");
  fs.createFile("/dest.txt", "dest content");

  const result = fs.rename("/source.txt", "/dest.txt");

  expect(result).toBe(false);
  expect(fs.exists("/source.txt")).toBe(true);
  expect(fs.readFile("/source.txt")).toBe("source content");
  expect(fs.readFile("/dest.txt")).toBe("dest content");
});

test("rename returns false when trying to rename root directory", () => {
  const fs = new VirtualFileSystem();

  const result = fs.rename("/", "/root");

  expect(result).toBe(false);
  expect(fs.exists("/")).toBe(true);
  expect(fs.exists("/root")).toBe(false);
});

test("rename returns false when destination is root", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  const result = fs.rename("/test.txt", "/");

  expect(result).toBe(false);
  expect(fs.exists("/test.txt")).toBe(true);
});

test("rename handles complex directory structure move", () => {
  const fs = new VirtualFileSystem();
  // Create complex structure
  fs.createDirectory("/project");
  fs.createDirectory("/project/src");
  fs.createDirectory("/project/src/components");
  fs.createDirectory("/project/src/utils");
  fs.createFile("/project/src/index.ts", "main");
  fs.createFile("/project/src/components/App.tsx", "app");
  fs.createFile("/project/src/components/Button.tsx", "button");
  fs.createFile("/project/src/utils/helpers.ts", "helpers");
  fs.createFile("/project/README.md", "readme");

  const result = fs.rename("/project", "/new-project");

  expect(result).toBe(true);
  expect(fs.exists("/project")).toBe(false);
  expect(fs.exists("/new-project/src/components/App.tsx")).toBe(true);
  expect(fs.exists("/new-project/src/utils/helpers.ts")).toBe(true);
  expect(fs.readFile("/new-project/src/index.ts")).toBe("main");
  expect(fs.readFile("/new-project/README.md")).toBe("readme");
});

test("rename preserves file content and directory structure", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/data");
  fs.createFile("/data/users.json", '{"users": []}');
  fs.createFile("/data/config.json", '{"version": "1.0"}');

  const result = fs.rename("/data", "/backup/2024/data");

  expect(result).toBe(true);
  expect(fs.exists("/backup")).toBe(true);
  expect(fs.exists("/backup/2024")).toBe(true);
  expect(fs.exists("/backup/2024/data")).toBe(true);
  expect(fs.readFile("/backup/2024/data/users.json")).toBe('{"users": []}');
  expect(fs.readFile("/backup/2024/data/config.json")).toBe('{"version": "1.0"}');
});

test("rename handles special characters in paths", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test file.txt", "content with spaces");

  const result = fs.rename("/test file.txt", "/docs/test-file.txt");

  expect(result).toBe(true);
  expect(fs.exists("/test file.txt")).toBe(false);
  expect(fs.exists("/docs/test-file.txt")).toBe(true);
  expect(fs.readFile("/docs/test-file.txt")).toBe("content with spaces");
});

test("rename normalizes paths correctly", () => {
  const fs = new VirtualFileSystem();
  fs.createFile("/test.txt", "content");

  const result = fs.rename("test.txt", "//new//path//file.txt");

  expect(result).toBe(true);
  expect(fs.exists("/test.txt")).toBe(false);
  expect(fs.exists("/new/path/file.txt")).toBe(true);
});

test("rename handles empty directories", () => {
  const fs = new VirtualFileSystem();
  fs.createDirectory("/empty-dir");

  const result = fs.rename("/empty-dir", "/moved-empty-dir");

  expect(result).toBe(true);
  expect(fs.exists("/empty-dir")).toBe(false);
  expect(fs.exists("/moved-empty-dir")).toBe(true);
  expect(fs.getNode("/moved-empty-dir")?.type).toBe("directory");
});
