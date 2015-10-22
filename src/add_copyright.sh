for i in `find . | grep "\.hpp$"` # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    cat copyright.txt $i >$i.new && mv $i.new $i
    echo $i
  fi
done


for i in `find . | grep "\.cpp$"` # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    cat copyright.txt $i >$i.new && mv $i.new $i
    echo $i
  fi
done

