#! /bin/bash -e
#
# replacement text for this commit
cat << EOF > commit.$$
This is a template for making commits to the git repository in this directory.
Lines with no 'at' characters, are general comments not connected to 
a specific file. Lines containing an 'at' character are "file name" 
followed by comment. Lines before the first 'at' character are preserved
during
	./commit.sh edit 
for example this entire paragraph is preserved.

commit.sh@ For this example, commit.sh is the only file committed.
EOF
# -----------------------------------------------------------------------------
if [ $0 != "./commit.sh" ]
then
	echo "./commit.sh: must be executed from directory containing commit.sh"
	rm commit.$$
	exit 1
fi
# -----------------------------------------------------------------------
if [ "$1" != 'list' ] && [ "$1" != 'edit' ] && [ "$1" != 'run' ]
then
cat << EOF
usage: ./commit.sh list
       ./commit.sh edit
       ./commit.sh run

list:
output a list of the files that have changes git knows about.

edit:
Edit the file list of files at the top of ./commit.sh to be the same as 
	./commit.sh list
would output. In addition, it displays the changes to ./commit.sh. This 
will include the new files in the list since the last edit of ./commit.sh. 
You should then edit ./commit.sh by hand, to add comments about the changes 
before running the command
	./commit.sh run

run:
commits changes to the list of files in ./commit.sh 
(provided that you reply y to the [y/n] prompt that ./commit.sh generates).
The file ./commit.sh cannot be commited this way; use
	svn commit -m "your log message" ./commit.sh
to commit this file.
EOF
	rm commit.$$
	exit 1
fi
# -----------------------------------------------------------------------
if [ "$1" == 'list' ] || [ "$1" == 'edit' ]
then
	# ------------------------------------------------
	unknown=`git status -s | sed -n \
		-e '/^[^?]/d' \
		-e 's/^?? */ /' \
		-e '/[/ ]junk/d' \
		-e '/[/ ]makefile.in/d' \
		-e '/[/ ]config.h.in/d' \
		-e '/.*\.am$/p'  \
		-e '/.*\.cpp$/p'  \
		-e '/.*\.hpp$/p'  \
		-e '/.*\.in$/p'  \
		-e '/.*\.omh$/p'  \
		-e '/.*\.py$/p'  \
		-e '/.*\.sh$/p'`
	msg="aborting because the following files are unknown to git"
	print_msg="no"
	for file in $unknown
	do
		if [ "$print_msg" == "no" ]
		then
			echo "commit.sh: $msg"
			print_msg="yes"
		fi
		echo $file
	done
	if [ "$print_msg" == "yes" ]
	then
		rm commit.$$
		exit 1
	fi
	# -------------------------------------------------
	git status | sed -n \
		-e '/^#\t*deleted:/p' \
		-e '/^#\t*modified:/p' \
		-e '/^#\t*renamed:/p' \
		-e '/^#\t*new file:/p' | \
			sed -e 's/^.*: *//' -e 's/ -> /\n/' -e '/^commit.sh$/d' | \
				sort -u > commit.$$
	# -------------------------------------------------
	if [ "$1" == 'list' ]
	then
		cat commit.$$
		rm commit.$$
		exit 0
	fi
	#
	echo "mv commit.sh commit.sh.old"
	      mv commit.sh commit.sh.old
	#
	echo "creating new commit.sh"
	sed -n -e '1,/@/p' commit.sh.old | sed -e '/@/d' > commit.sh
	sed commit.$$ -e 's/$/@/'                       >> commit.sh
	sed -n -e '/^EOF/,$p' commit.sh.old             >> commit.sh
	rm  commit.$$
	#
	echo "------------------------------------"
	echo "diff commit.sh.old commit.sh"
	if diff    commit.sh.old commit.sh
	then
		echo "commit.sh edit: no changes to commit.sh"
	fi
	echo "------------------------------------"
     echo "chmod +x commit.sh"
           chmod +x commit.sh
	#
	exit 0
fi
# -----------------------------------------------------------------------
list=`sed -e '/@/! d' -e 's/@.*//' commit.$$`
msg=`sed -e '/@ *$/d' -e 's|.*/\([^/]*@\)|\1|' -e 's|@|:|' commit.$$`
if (echo $list | grep '^commit.sh$' > /dev/null)
then
	echo "commit.sh: cannot be used to commit changes to itself."
	echo "remove it from the list of files in commit.sh"
	exit 1
fi
rm commit.$$
echo "giit commit -m \""
echo "$msg"
echo "\" \\"
echo "$list"
read -p "is this ok [y/n] ?" response
if [ "$response" != "y" ]
then
	exit 1
fi
#
if ! git commit -m "$msg" $list
then
	echo "commit.sh: commit failed"
	exit 1
fi
#
echo "mv commit.sh commit.sh.old"
      mv commit.sh commit.sh.old
#
echo "git checkout commit.sh"
      git checkout commit.sh
