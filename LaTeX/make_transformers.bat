@echo off
for /r %%i in (Main*Transformer*.tex) do texify -cp %%i
