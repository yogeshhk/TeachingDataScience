@echo off
for /r %%i in (Main_*SeqSeg*.tex) do texify -cp %%i
