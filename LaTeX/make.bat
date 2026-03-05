@echo off
for /r %%i in (Main_Seminar_AI_BizLeaders*.tex) do texify -cp %%i
