@echo off
for /r %%i in (Main_*GNN*.tex) do texify -cp %%i
