import { test, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { FileTree } from "@/components/editor/FileTree";
import {
  type VirtualFileSystem as FileSystem,
  FileNode,
} from "@/lib/file-system";
import { useFileSystem } from "@/lib/contexts/file-system-context";

// Mock the file system context
vi.mock("@/lib/contexts/file-system-context");

// Clean up after each test
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// Mock lucide-react icons
vi.mock("lucide-react", () => ({
  ChevronRight: ({ className }: { className?: string }) => (
    <div className={className}>ChevronRight</div>
  ),
  ChevronDown: ({ className }: { className?: string }) => (
    <div className={className}>ChevronDown</div>
  ),
  Folder: ({ className }: { className?: string }) => (
    <div className={className}>Folder</div>
  ),
  FolderOpen: ({ className }: { className?: string }) => (
    <div className={className}>FolderOpen</div>
  ),
  FileCode: ({ className }: { className?: string }) => (
    <div className={className}>FileCode</div>
  ),
}));

// Helper function to create a mock file system
function createMockFileSystem(nodes: Record<string, FileNode>) {
  const mockFileSystem = {
    getNode: (path: string) => nodes[path],
  };
  return mockFileSystem as FileSystem;
}

test("FileTree renders empty state when no files exist", () => {
  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: new Map() },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  render(<FileTree />);

  expect(screen.getByText("No files yet")).toBeDefined();
  expect(screen.getByText("Files will appear here")).toBeDefined();
});

test("FileTree renders files and directories", () => {
  const rootChildren = new Map<string, FileNode>([
    [
      "components",
      {
        type: "directory",
        name: "components",
        path: "/components",
        children: new Map(),
      },
    ],
    [
      "App.jsx",
      { type: "file", name: "App.jsx", path: "/App.jsx", content: "" },
    ],
  ]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  render(<FileTree />);

  expect(screen.getByText("components")).toBeDefined();
  expect(screen.getByText("App.jsx")).toBeDefined();
});

test("FileTree sorts directories before files", () => {
  const rootChildren = new Map<string, FileNode>([
    ["b.txt", { type: "file", name: "b.txt", path: "/b.txt", content: "" }],
    [
      "a-folder",
      {
        type: "directory",
        name: "a-folder",
        path: "/a-folder",
        children: new Map(),
      },
    ],
    ["c.js", { type: "file", name: "c.js", path: "/c.js", content: "" }],
    [
      "z-folder",
      {
        type: "directory",
        name: "z-folder",
        path: "/z-folder",
        children: new Map(),
      },
    ],
  ]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  render(<FileTree />);

  const items = screen.getAllByText(/^(a-folder|z-folder|b\.txt|c\.js)$/);
  const itemTexts = items.map((item) => item.textContent);

  // Directories should come first (alphabetically), then files (alphabetically)
  expect(itemTexts).toEqual(["a-folder", "z-folder", "b.txt", "c.js"]);
});

test("FileTreeNode expands and collapses directories", () => {
  const childNode: FileNode = {
    type: "file",
    name: "child.txt",
    path: "/parent/child.txt",
    content: "",
  };
  const parentChildren = new Map([["child.txt", childNode]]);
  const parentNode: FileNode = {
    type: "directory",
    name: "parent",
    path: "/parent",
    children: parentChildren,
  };
  const rootChildren = new Map([["parent", parentNode]]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
    "/parent": parentNode,
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  const { container } = render(<FileTree />);

  // Initially expanded - should show child
  expect(screen.getByText("parent")).toBeDefined();
  expect(screen.getByText("child.txt")).toBeDefined();

  // Find the chevron icon next to "parent" directory
  const parentDiv = screen.getByText("parent").parentElement;
  const chevronDown = parentDiv?.querySelector(".h-3\\.5.w-3\\.5");
  expect(chevronDown?.textContent).toBe("ChevronDown");

  // Click to collapse
  fireEvent.click(screen.getByText("parent"));

  // Child should be hidden, chevron should point right
  expect(screen.queryByText("child.txt")).toBeNull();
  const chevronRight = parentDiv?.querySelector(".h-3\\.5.w-3\\.5");
  expect(chevronRight?.textContent).toBe("ChevronRight");
});

test("FileTreeNode selects file when clicked", () => {
  const mockSetSelectedFile = vi.fn();
  const fileNode: FileNode = {
    type: "file",
    name: "test.js",
    path: "/test.js",
    content: "",
  };
  const rootChildren = new Map([["test.js", fileNode]]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: mockSetSelectedFile,
  });

  render(<FileTree />);

  fireEvent.click(screen.getByText("test.js"));

  expect(mockSetSelectedFile).toHaveBeenCalledWith("/test.js");
});

test("FileTreeNode highlights selected file", () => {
  const fileNode: FileNode = {
    type: "file",
    name: "selected.js",
    path: "/selected.js",
    content: "",
  };
  const rootChildren = new Map([["selected.js", fileNode]]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: "/selected.js",
    setSelectedFile: vi.fn(),
  });

  const { container } = render(<FileTree />);

  // Find the div containing the file name
  const fileDiv = screen.getByText("selected.js").parentElement;
  expect(fileDiv?.className).toContain("bg-blue-50");
  expect(fileDiv?.className).toContain("text-blue-600");
});

test("FileTree renders nested directory structure", () => {
  const deepFile: FileNode = {
    type: "file",
    name: "deep.txt",
    path: "/a/b/c/deep.txt",
    content: "",
  };
  const cChildren = new Map([["deep.txt", deepFile]]);
  const cNode: FileNode = {
    type: "directory",
    name: "c",
    path: "/a/b/c",
    children: cChildren,
  };
  const bChildren = new Map([["c", cNode]]);
  const bNode: FileNode = {
    type: "directory",
    name: "b",
    path: "/a/b",
    children: bChildren,
  };
  const aChildren = new Map([["b", bNode]]);
  const aNode: FileNode = {
    type: "directory",
    name: "a",
    path: "/a",
    children: aChildren,
  };
  const rootChildren = new Map([["a", aNode]]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 0,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  render(<FileTree />);

  expect(screen.getByText("a")).toBeDefined();
  expect(screen.getByText("b")).toBeDefined();
  expect(screen.getByText("c")).toBeDefined();
  expect(screen.getByText("deep.txt")).toBeDefined();
});

test("FileTree re-renders when refreshTrigger changes", () => {
  const fileNode: FileNode = {
    type: "file",
    name: "test.js",
    path: "/test.js",
    content: "",
  };
  const rootChildren = new Map([["test.js", fileNode]]);

  const mockFileSystem = createMockFileSystem({
    "/": { type: "directory", name: "", path: "/", children: rootChildren },
  });

  const mockUseFileSystem = useFileSystem as ReturnType<typeof vi.fn>;
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 1,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  const { rerender } = render(<FileTree />);

  // Update with new refreshTrigger
  mockUseFileSystem.mockReturnValue({
    fileSystem: mockFileSystem,
    refreshTrigger: 2,
    selectedFile: null,
    setSelectedFile: vi.fn(),
  });

  rerender(<FileTree />);

  // The component should still render correctly
  expect(screen.getByText("test.js")).toBeDefined();
});
