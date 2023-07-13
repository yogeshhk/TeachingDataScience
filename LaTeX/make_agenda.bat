@echo off
for /r %%i in (Main_Agenda*.tex) do texify -cp %%i
