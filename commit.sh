#! /bin/bash
#
# replace text in msg for this commit
msg="summary_commnet_ending_with_semicolon;

file_to_commit:comment_for_this_file_ending_with_semicolon;
...
file_to_commit:comment_for_this_file_ending_with_semicolon;
"
# -----------------------------------------------------------------------
list=`echo $msg | sed -e 's|^[^;]*;||' -e 's|:[^;]*;||g'`
msg=`echo $msg | sed -e 's|omh/||g' -e 's|pycppad/||g' \
	-e 's|example/||g' -e 's|;|.\n|g'` 
echo "git commit \\"
echo "\"$msg\" \\"
echo "$list"
read -p "is this ok [y/n] ?" response
if [ "$response" != "y" ]
then
	exit 1
fi
git commit -m "$msg" $list
