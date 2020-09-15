@echo off
for /r %%i in (Main_Seminar_Word*.tex) do texify -cp %%i
