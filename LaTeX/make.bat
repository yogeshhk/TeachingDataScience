@echo off
for /r %%i in (Main_Seminar_LLM_Intro*.tex) do texify -cp %%i
