@echo off
for /r %%i in (Main_Seminar_LLM_RAG*.tex) do texify -cp %%i
