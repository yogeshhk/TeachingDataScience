@echo off
for /r %%i in (Main_*_ChatGPT*.tex) do texify -cp %%i
