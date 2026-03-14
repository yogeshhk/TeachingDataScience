@echo off
for /r %%i in (Main_Seminar_*LLM_RAG*.tex) do texify -cp %%i
