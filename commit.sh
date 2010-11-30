#! /bin/bash
#
set -e
#
# replacement text for this commit
cat << EOF > commit.$$
This is a template file for making commits to the pycppad repository.
Lines with no 'at' characters, are general comments not connected to 
a specific file. Lines containing an 'at' character are "file name" 
followed by comment; for example

commit.sh@ For this example, commit.sh is the only file committed.
EOF
# -----------------------------------------------------------------------
if [ "$1" == 'files' ]
then
	echo "rm commit.[0-9]*"
	rm commit.[0-9]*
	#
	echo "cp commit.sh commit.old"
	cp commit.sh commit.old
	#
	echo "git checkout commit.sh"
	git checkout commit.sh
	#
	# section of commit.sh above and including first empty line
	sed -n -e '1,/^$/p' < commit.sh > commit.$$
	#
	# list of files that need to be committed
	git status | sed -n \
		-e '/^#\t*deleted:/p' \
		-e '/^#\t*modified:/p' \
		-e '/^#\t*renamed:/p' \
		-e '/^#\t*new file:/p' | sed \
			-e 's/^.*: *//' \
			-e 's/$/@/' \
			-e 's/ -> /@\n/' | sort -u >> commit.$$
	#
	# print list of files
	echo "FILES:"
	sed -e '1,/^$/d' < commit.$$
	echo ""
	#
	# section of commit.sh below and including first EOF at line beginning
	sed -n -e '/^EOF/,$p' < commit.sh >> commit.$$
	#
	echo "mv commit.$$ commit.sh"
	mv commit.$$ commit.sh
	#
	chmod +x commit.sh
	exit 0
fi
# -----------------------------------------------------------------------
if [ "$1" != 'run' ]
then
cat << EOF
usage: ./commit.sh files
       ./commit.sh run

The first from changes the list of files at the beginning of commit.sh 
so that it contains all the files that have changed status.
You should then edit commit.sh by hand (as per the instrucgtion at the 
beginning of commit.sh) before running the second form.

The second form actually commits the list of files (provided that you reply
y to the [y/n] prompt that commit.sh generates).
EOF
	rm commit.$$
	exit 0
fi
# -----------------------------------------------------------------------
list=`sed -e '/@/! d' -e 's/@.*//' commit.$$`
msg=`sed -e '/@ *$/d' -e 's|.*/\([^/]*@\)|\1|' -e 's|@|:|' commit.$$`
rm commit.$$
echo "git commit -m \""
echo "$msg"
echo "\" \\"
echo "$list"
read -p "is this ok [y/n] ?" response
if [ "$response" != "y" ]
then
     exit 1
fi
#
echo "git commit -m '$msg' $list"
if ! git commit -m "$msg" $list
then
	echo "commit.sh: failed"
	exit 1
fi
#
echo "mv commit.sh commit.old"
mv commit.sh commit.old
#
echo "git checkout commit.sh"
git checkout commit.sh
