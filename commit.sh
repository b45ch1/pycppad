#! /bin/bash
#
# replacement text for this commit
cat << EOF > commit.$$
Modify commit.sh to use a temporary file for the message.

Lines with colon characters are file names followed by messages; for example,
commit.sh: in this case, commit.sh is the file to be commited.
EOF
# -----------------------------------------------------------------------
list=`sed -e '/:/! d' -e 's/:.*//' commit.$$`
msg=`sed -e 's|.*/\([^/]*:\)|\1|' commit.$$` 
rm commit.$$
echo "git commit \\"
echo "\"$msg\" \\"
echo "\\"
echo "$list"
read -p "is this ok [y/n] ?" response
if [ "$response" != "y" ]
then
	exit 1
fi
git commit -m "$msg" $list
