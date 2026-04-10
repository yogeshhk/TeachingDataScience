import { tool } from "ai";
import { z } from "zod";
import { VirtualFileSystem } from "../file-system";

export function buildFileManagerTool(fileSystem: VirtualFileSystem) {
  return tool({
    description:
      'Rename or delete files or folders in the file system. Rename can be used to "move" a file. Rename will recursively create folders as required.',
    parameters: z.object({
      command: z
        .enum(["rename", "delete"])
        .describe("The operation to perform"),
      path: z
        .string()
        .describe("The path to the file or directory to rename or delete"),
      new_path: z
        .string()
        .optional()
        .describe("The new path. Only provide when renaming or moving a file."),
    }),
    execute: async ({ command, path, new_path }) => {
      if (command === "rename") {
        if (!new_path) {
          return {
            success: false,
            error: "new_path is required for rename command",
          };
        }
        const success = fileSystem.rename(path, new_path);
        if (success) {
          return {
            success: true,
            message: `Successfully renamed ${path} to ${new_path}`,
          };
        } else {
          return {
            success: false,
            error: `Failed to rename ${path} to ${new_path}`,
          };
        }
      } else if (command === "delete") {
        const success = fileSystem.deleteFile(path);
        if (success) {
          return { success: true, message: `Successfully deleted ${path}` };
        } else {
          return { success: false, error: `Failed to delete ${path}` };
        }
      }

      return { success: false, error: "Invalid command" };
    },
  });
}
