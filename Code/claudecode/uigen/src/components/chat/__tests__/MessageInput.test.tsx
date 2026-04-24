import { test, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MessageInput } from "../MessageInput";

afterEach(() => {
  cleanup();
});

test("renders with placeholder text", () => {
  const mockProps = {
    input: "",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByPlaceholderText("Describe the React component you want to create...");
  expect(textarea).toBeDefined();
});

test("displays the input value", () => {
  const mockProps = {
    input: "Test input value",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByDisplayValue("Test input value");
  expect(textarea).toBeDefined();
});

test("calls handleInputChange when typing", async () => {
  const handleInputChange = vi.fn();
  const mockProps = {
    input: "",
    handleInputChange,
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByPlaceholderText("Describe the React component you want to create...");
  await userEvent.type(textarea, "Hello");
  
  expect(handleInputChange).toHaveBeenCalled();
});

test("calls handleSubmit when form is submitted", async () => {
  const handleSubmit = vi.fn((e) => e.preventDefault());
  const mockProps = {
    input: "Test input",
    handleInputChange: vi.fn(),
    handleSubmit,
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const form = screen.getByRole("textbox").closest("form")!;
  fireEvent.submit(form);
  
  expect(handleSubmit).toHaveBeenCalledOnce();
});

test("submits form when Enter is pressed without shift", async () => {
  const handleSubmit = vi.fn((e) => e.preventDefault());
  const mockProps = {
    input: "Test input",
    handleInputChange: vi.fn(),
    handleSubmit,
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByRole("textbox");
  fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
  
  expect(handleSubmit).toHaveBeenCalledOnce();
});

test("does not submit form when Enter is pressed with shift", async () => {
  const handleSubmit = vi.fn((e) => e.preventDefault());
  const mockProps = {
    input: "Test input",
    handleInputChange: vi.fn(),
    handleSubmit,
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByRole("textbox");
  fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
  
  expect(handleSubmit).not.toHaveBeenCalled();
});

test("disables textarea when isLoading is true", () => {
  const mockProps = {
    input: "",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: true,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByRole("textbox");
  expect(textarea).toHaveProperty("disabled", true);
});

test("disables submit button when isLoading is true", () => {
  const mockProps = {
    input: "Test input",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: true,
  };

  render(<MessageInput {...mockProps} />);
  
  const submitButton = screen.getByRole("button");
  expect(submitButton).toHaveProperty("disabled", true);
});

test("disables submit button when input is empty", () => {
  const mockProps = {
    input: "",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const submitButton = screen.getByRole("button");
  expect(submitButton).toHaveProperty("disabled", true);
});

test("disables submit button when input contains only whitespace", () => {
  const mockProps = {
    input: "   ",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const submitButton = screen.getByRole("button");
  expect(submitButton).toHaveProperty("disabled", true);
});

test("enables submit button when input has content and not loading", () => {
  const mockProps = {
    input: "Valid content",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const submitButton = screen.getByRole("button");
  expect(submitButton).toHaveProperty("disabled", false);
});

test("applies correct CSS classes based on loading state", () => {
  const { rerender } = render(
    <MessageInput
      input="Test"
      handleInputChange={vi.fn()}
      handleSubmit={vi.fn()}
      isLoading={false}
    />
  );

  let submitButton = screen.getByRole("button");
  expect(submitButton.className).toContain("disabled:opacity-40");
  expect(submitButton.className).toContain("hover:bg-blue-50");

  rerender(
    <MessageInput
      input="Test"
      handleInputChange={vi.fn()}
      handleSubmit={vi.fn()}
      isLoading={true}
    />
  );

  submitButton = screen.getByRole("button");
  expect(submitButton.className).toContain("disabled:cursor-not-allowed");
  expect(submitButton.className).toContain("disabled:opacity-40");
});

test("applies pulse animation to send icon when loading", () => {
  const { rerender } = render(
    <MessageInput
      input="Test"
      handleInputChange={vi.fn()}
      handleSubmit={vi.fn()}
      isLoading={false}
    />
  );

  let sendIcon = screen.getByRole("button").querySelector("svg");
  expect(sendIcon?.getAttribute("class")).not.toContain("animate-pulse");

  rerender(
    <MessageInput
      input="Test"
      handleInputChange={vi.fn()}
      handleSubmit={vi.fn()}
      isLoading={true}
    />
  );

  sendIcon = screen.getByRole("button").querySelector("svg");
  expect(sendIcon?.getAttribute("class")).toContain("text-neutral-300");
});

test("textarea has correct styling classes", () => {
  const mockProps = {
    input: "",
    handleInputChange: vi.fn(),
    handleSubmit: vi.fn(),
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const textarea = screen.getByRole("textbox");
  expect(textarea.className).toContain("min-h-[80px]");
  expect(textarea.className).toContain("max-h-[200px]");
  expect(textarea.className).toContain("resize-none");
  expect(textarea.className).toContain("focus:ring-2");
  expect(textarea.className).toContain("focus:ring-blue-500/10");
});

test("submit button click triggers form submission", async () => {
  const handleSubmit = vi.fn((e) => e.preventDefault());
  const mockProps = {
    input: "Test input",
    handleInputChange: vi.fn(),
    handleSubmit,
    isLoading: false,
  };

  render(<MessageInput {...mockProps} />);
  
  const submitButton = screen.getByRole("button");
  await userEvent.click(submitButton);
  
  expect(handleSubmit).toHaveBeenCalledOnce();
});