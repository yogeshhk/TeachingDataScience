@echo off
for /r %%i in (Main_Seminar_*LLM_Agents*.tex) do texify -cp %%i
