@echo off
for /r %%i in (Main_Seminar_LLM_Agents*.tex) do texify -cp %%i
