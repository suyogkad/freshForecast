@echo off
echo Pushing changes to GitHub...

git status
git add .
git status
git commit -m "Project Updated"
git push

echo Done!
exit
