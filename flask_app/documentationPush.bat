@echo off
echo Pushing to GitHub..

git status 
git add .
git status
git commit -m "Documentation Updates"
git push

echo Exit
echo Done!