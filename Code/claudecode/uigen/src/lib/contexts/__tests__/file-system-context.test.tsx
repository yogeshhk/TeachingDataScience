import { test, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";
import { FileSystemProvider, useFileSystem } from "../file-system-context";
import { VirtualFileSystem } from "@/lib/file-system";

// Mock the VirtualFileSystem
vi.mock("@/lib/file-system", () => ({
  VirtualFileSystem: vi.fn(),
}));

const mockFileSystem = {
  createFile: vi.fn(),
  updateFile: vi.fn(),
  deleteFile: vi.fn(),
  rename: vi.fn(),
  readFile: vi.fn(),
  getAllFiles: vi.fn(),
  createFileWithParents: vi.fn(),
  replaceInFile: vi.fn(),
  insertInFile: vi.fn(),
  getNode: vi.fn(),
  serialize: vi.fn(() => ({})),
};

beforeEach(() => {
  vi.clearAllMocks();
  (VirtualFileSystem as any).mockImplementation(() => mockFileSystem);
  mockFileSystem.getAllFiles.mockReturnValue(new Map());
});

afterEach(() => {
  cleanup();
});

test("provides file system methods", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  expect(result.current.fileSystem).toBeDefined();
  expect(result.current.selectedFile).toBeNull();
  expect(result.current.createFile).toBeDefined();
  expect(result.current.updateFile).toBeDefined();
  expect(result.current.deleteFile).toBeDefined();
  expect(result.current.renameFile).toBeDefined();
  expect(result.current.getFileContent).toBeDefined();
  expect(result.current.getAllFiles).toBeDefined();
});

test("createFile calls fileSystem method and triggers refresh", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.createFile("/test.js", "content");
  });

  expect(mockFileSystem.createFile).toHaveBeenCalledWith("/test.js", "content");
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("updateFile calls fileSystem method and triggers refresh", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.updateFile("/test.js", "new content");
  });

  expect(mockFileSystem.updateFile).toHaveBeenCalledWith("/test.js", "new content");
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("deleteFile calls fileSystem method and triggers refresh", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.deleteFile("/test.js");
  });

  expect(mockFileSystem.deleteFile).toHaveBeenCalledWith("/test.js");
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("deleteFile clears selectedFile if it matches deleted file", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.setSelectedFile("/test.js");
  });

  expect(result.current.selectedFile).toBe("/test.js");

  act(() => {
    result.current.deleteFile("/test.js");
  });

  expect(result.current.selectedFile).toBeNull();
});

test("renameFile updates selectedFile when renaming selected file", () => {
  mockFileSystem.rename.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.setSelectedFile("/old.js");
  });

  act(() => {
    result.current.renameFile("/old.js", "/new.js");
  });

  expect(mockFileSystem.rename).toHaveBeenCalledWith("/old.js", "/new.js");
  expect(result.current.selectedFile).toBe("/new.js");
});

test("renameFile updates selectedFile when file is inside renamed directory", () => {
  mockFileSystem.rename.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.setSelectedFile("/old-dir/file.js");
  });

  act(() => {
    result.current.renameFile("/old-dir", "/new-dir");
  });

  expect(result.current.selectedFile).toBe("/new-dir/file.js");
});

test("renameFile returns false when operation fails", () => {
  mockFileSystem.rename.mockReturnValue(false);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    const success = result.current.renameFile("/old.js", "/new.js");
    expect(success).toBe(false);
  });

  expect(result.current.refreshTrigger).toBe(initialTrigger); // No refresh on failure
});

test("getFileContent calls fileSystem readFile", () => {
  mockFileSystem.readFile.mockReturnValue("file content");

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const content = result.current.getFileContent("/test.js");

  expect(mockFileSystem.readFile).toHaveBeenCalledWith("/test.js");
  expect(content).toBe("file content");
});

test("getAllFiles calls fileSystem getAllFiles", () => {
  const filesMap = new Map([
    ["/file1.js", "content1"],
    ["/file2.js", "content2"],
  ]);
  mockFileSystem.getAllFiles.mockReturnValue(filesMap);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const files = result.current.getAllFiles();

  expect(mockFileSystem.getAllFiles).toHaveBeenCalled();
  expect(files).toBe(filesMap);
});

test("selects App.jsx by default if it exists", () => {
  const filesMap = new Map([
    ["/App.jsx", "content"],
    ["/other.js", "content"],
  ]);
  mockFileSystem.getAllFiles.mockReturnValue(filesMap);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  expect(result.current.selectedFile).toBe("/App.jsx");
});

test("selects first root file when App.jsx doesn't exist", () => {
  const filesMap = new Map([
    ["/b.js", "content"],
    ["/a.js", "content"],
    ["/nested/file.js", "content"],
  ]);
  mockFileSystem.getAllFiles.mockReturnValue(filesMap);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  expect(result.current.selectedFile).toBe("/a.js"); // First file alphabetically
});

test("throws error when used outside provider", () => {
  expect(() => {
    renderHook(() => useFileSystem());
  }).toThrowError("useFileSystem must be used within a FileSystemProvider");
});

test("uses provided file system when passed", () => {
  const customFileSystem = {
    ...mockFileSystem,
    custom: true,
  };

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => (
      <FileSystemProvider fileSystem={customFileSystem as any}>
        {children}
      </FileSystemProvider>
    ),
  });

  expect(result.current.fileSystem).toBe(customFileSystem);
});

// Tool call tests
test("handles str_replace_editor create command", () => {
  mockFileSystem.createFileWithParents.mockReturnValue("File created");
  mockFileSystem.createFile.mockReturnValue({});

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "create",
        path: "/test.js",
        file_text: "console.log('test');",
      },
    });
  });

  expect(mockFileSystem.createFileWithParents).toHaveBeenCalledWith(
    "/test.js",
    "console.log('test');"
  );
  expect(mockFileSystem.createFile).toHaveBeenCalledWith(
    "/test.js",
    "console.log('test');"
  );
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("handles str_replace_editor create command with error", () => {
  mockFileSystem.createFileWithParents.mockReturnValue("Error: File exists");

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "create",
        path: "/test.js",
        file_text: "content",
      },
    });
  });

  expect(result.current.refreshTrigger).toBe(initialTrigger); // No refresh on error
});

test("handles str_replace_editor str_replace command", () => {
  mockFileSystem.replaceInFile.mockReturnValue("Replaced successfully");
  mockFileSystem.readFile.mockReturnValue("new content");
  mockFileSystem.updateFile.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "str_replace",
        path: "/test.js",
        old_str: "old",
        new_str: "new",
      },
    });
  });

  expect(mockFileSystem.replaceInFile).toHaveBeenCalledWith("/test.js", "old", "new");
  expect(mockFileSystem.readFile).toHaveBeenCalledWith("/test.js");
  expect(mockFileSystem.updateFile).toHaveBeenCalledWith("/test.js", "new content");
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("handles str_replace_editor str_replace command with error", () => {
  mockFileSystem.replaceInFile.mockReturnValue("Error: Not found");

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "str_replace",
        path: "/test.js",
        old_str: "old",
        new_str: "new",
      },
    });
  });

  expect(mockFileSystem.readFile).not.toHaveBeenCalled();
  expect(mockFileSystem.updateFile).not.toHaveBeenCalled();
});

test("handles str_replace_editor insert command", () => {
  mockFileSystem.insertInFile.mockReturnValue("Inserted successfully");
  mockFileSystem.readFile.mockReturnValue("updated content");
  mockFileSystem.updateFile.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  const initialTrigger = result.current.refreshTrigger;

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "insert",
        path: "/test.js",
        new_str: "new line",
        insert_line: 5,
      },
    });
  });

  expect(mockFileSystem.insertInFile).toHaveBeenCalledWith("/test.js", 5, "new line");
  expect(mockFileSystem.readFile).toHaveBeenCalledWith("/test.js");
  expect(mockFileSystem.updateFile).toHaveBeenCalledWith("/test.js", "updated content");
  expect(result.current.refreshTrigger).toBe(initialTrigger + 1);
});

test("handles str_replace_editor insert command with error", () => {
  mockFileSystem.insertInFile.mockReturnValue("Error: Invalid line");

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "insert",
        path: "/test.js",
        new_str: "new line",
        insert_line: 999,
      },
    });
  });

  expect(mockFileSystem.readFile).not.toHaveBeenCalled();
  expect(mockFileSystem.updateFile).not.toHaveBeenCalled();
});

test("handles file_manager rename command", () => {
  mockFileSystem.rename.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.handleToolCall({
      toolName: "file_manager",
      args: {
        command: "rename",
        path: "/old.js",
        new_path: "/new.js",
      },
    });
  });

  expect(mockFileSystem.rename).toHaveBeenCalledWith("/old.js", "/new.js");
});

test("handles file_manager delete command", () => {
  mockFileSystem.deleteFile.mockReturnValue(true);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.handleToolCall({
      toolName: "file_manager",
      args: {
        command: "delete",
        path: "/test.js",
      },
    });
  });

  expect(mockFileSystem.deleteFile).toHaveBeenCalledWith("/test.js");
});

test("handles unknown tool name gracefully", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  expect(() => {
    act(() => {
      result.current.handleToolCall({
        toolName: "unknown_tool",
        args: {},
      });
    });
  }).not.toThrow();
});

test("handles unknown command gracefully", () => {
  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  expect(() => {
    act(() => {
      result.current.handleToolCall({
        toolName: "str_replace_editor",
        args: {
          command: "unknown",
          path: "/test.js",
        },
      });
    });
  }).not.toThrow();

  expect(mockFileSystem.createFileWithParents).not.toHaveBeenCalled();
  expect(mockFileSystem.replaceInFile).not.toHaveBeenCalled();
  expect(mockFileSystem.insertInFile).not.toHaveBeenCalled();
});

test("handles null file content when updating file", () => {
  mockFileSystem.replaceInFile.mockReturnValue("Replaced successfully");
  mockFileSystem.readFile.mockReturnValue(null);

  const { result } = renderHook(() => useFileSystem(), {
    wrapper: ({ children }) => <FileSystemProvider>{children}</FileSystemProvider>,
  });

  act(() => {
    result.current.handleToolCall({
      toolName: "str_replace_editor",
      args: {
        command: "str_replace",
        path: "/test.js",
        old_str: "old",
        new_str: "new",
      },
    });
  });

  expect(mockFileSystem.updateFile).not.toHaveBeenCalled();
});