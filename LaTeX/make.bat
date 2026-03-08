@echo off
for /r %%i in (Main_Seminar_LLM_PromptEngg*.tex) do texify -cp %%i
