echo "Which branch do you want to run your program? Press ENTER if you want to use 'HEAD'."
read GIT_REF
if [ "$GIT_REF" == "" ]
  then
    GIT_REF="HEAD"
    echo "Using ${GIT_REF}"
  else
    echo "Using ${GIT_REF} branch"
fi

