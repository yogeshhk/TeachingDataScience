"use server";

import { getSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

interface CreateProjectInput {
  name: string;
  messages: any[];
  data: Record<string, any>;
}

export async function createProject(input: CreateProjectInput) {
  const session = await getSession();
  
  if (!session) {
    throw new Error("Unauthorized");
  }

  const project = await prisma.project.create({
    data: {
      name: input.name,
      userId: session.userId,
      messages: JSON.stringify(input.messages),
      data: JSON.stringify(input.data),
    },
  });

  return project;
}