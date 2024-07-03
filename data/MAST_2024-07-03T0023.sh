#!/bin/bash
#
# Requires bash version >= 4
# 
# The script uses the command line tool 'curl' for querying
# the MAST Download service for public and protected data. 
#

type -P curl >&/dev/null || { echo "This script requires curl. Exiting." >&2; exit 1; }



# Check for existing Download Folder
# prompt user for overwrite, if found
let EXTENSION=0
FOLDER=MAST_2024-07-03T0023
DOWNLOAD_FOLDER=$FOLDER
if [ -d $DOWNLOAD_FOLDER ]
then
  echo -n "Download folder ${DOWNLOAD_FOLDER} found, (C)ontinue writing to existing folder or use (N)ew folder? [N]> "
  read -n1 ans
  if [ "$ans" = "c" -o "$ans" = "C" ]
  then
    echo ""
    echo "Downloading to existing folder: ${DOWNLOAD_FOLDER}"
    CONT="-C -"
  else
    while [ -d $DOWNLOAD_FOLDER ]
    do
      ((EXTENSION++))
      DOWNLOAD_FOLDER="${FOLDER}-${EXTENSION}"
    done

    echo ""
    echo "Downloading to new folder: ${DOWNLOAD_FOLDER}"
  fi
fi

# mkdir if it doesn't exist and download files there. 
mkdir -p ${DOWNLOAD_FOLDER}

cat >${DOWNLOAD_FOLDER}/MANIFEST.HTML<<EOT
<!DOCTYPE html>
<html>
    <head>
        <title>MAST_2024-07-03T0023</title>
    </head>
    <body>
        <h2>Manifest for File: MAST_2024-07-03T0023</h2>
        <h3>Total Files: 589</h3>
        <table cellspacing="0" cellpadding="4" rules="all" style="border-width:5px; border-style:solid; border-collapse:collapse;">
            <tr>
                <td><b>URI</b></td>
                <td><b>File</b></td>
                <td><b>Access</b></td>
                <td><b>Status</b></td>
                <td><b>Logged In User</b></td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63010_asc.fits</td>
                <td>HST/n8yj63010/n8yj63010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66010_asn.fits</td>
                <td>HST/n8yj66010/n8yj66010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65010_asn.fits</td>
                <td>HST/n8yj65010/n8yj65010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10010_mos.fits</td>
                <td>HST/n9nk10010/n9nk10010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28010_mos.fits</td>
                <td>HST/n9nk28010/n9nk28010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68020_asn.fits</td>
                <td>HST/n8yj68020/n8yj68020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06010_asc.fits</td>
                <td>HST/n8yj06010/n8yj06010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09010_mos.fits</td>
                <td>HST/n8yj09010/n8yj09010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10010_asc.fits</td>
                <td>HST/n9nk10010/n9nk10010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29010_asn.fits</td>
                <td>HST/n8yj29010/n8yj29010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32010_asc.fits</td>
                <td>HST/n8yj32010/n8yj32010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24010_mos.fits</td>
                <td>HST/n8yj24010/n8yj24010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13020_asn.fits</td>
                <td>HST/n8yj13020/n8yj13020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43020_asc.fits</td>
                <td>HST/n8yj43020/n8yj43020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59020_mos.fits</td>
                <td>HST/n8yj59020/n8yj59020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55010_mos.fits</td>
                <td>HST/n8yj55010/n8yj55010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13010_asn.fits</td>
                <td>HST/n8yj13010/n8yj13010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45010_asn.fits</td>
                <td>HST/n8yj45010/n8yj45010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53010_asn.fits</td>
                <td>HST/n8yj53010/n8yj53010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10020_asn.fits</td>
                <td>HST/n9nk10020/n9nk10020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14010_asc.fits</td>
                <td>HST/n8yj14010/n8yj14010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56010_asc.fits</td>
                <td>HST/n8yj56010/n8yj56010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44010_mos.fits</td>
                <td>HST/n8yj44010/n8yj44010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25010_mos.fits</td>
                <td>HST/n8yj25010/n8yj25010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05020_asc.fits</td>
                <td>HST/n8yj05020/n8yj05020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10020_mos.fits</td>
                <td>HST/n8yj10020/n8yj10020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31020_mos.fits</td>
                <td>HST/n8yj31020/n8yj31020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30010_mos.fits</td>
                <td>HST/n9nk30010/n9nk30010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46010_asc.fits</td>
                <td>HST/n8yj46010/n8yj46010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31020_asn.fits</td>
                <td>HST/n9nk31020/n9nk31020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31020_asc.fits</td>
                <td>HST/n9nk31020/n9nk31020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63020_asc.fits</td>
                <td>HST/n8yj63020/n8yj63020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27020_mos.fits</td>
                <td>HST/n9nk27020/n9nk27020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26020_mos.fits</td>
                <td>HST/n8yj26020/n8yj26020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10020_asc.fits</td>
                <td>HST/n8yj10020/n8yj10020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30020_mos.fits</td>
                <td>HST/n9nk30020/n9nk30020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25020_asn.fits</td>
                <td>HST/n8yj25020/n8yj25020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54010_mos.fits</td>
                <td>HST/n8yj54010/n8yj54010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31020_mos.fits</td>
                <td>HST/n9nk31020/n9nk31020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53010_asc.fits</td>
                <td>HST/n8yj53010/n8yj53010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20010_asc.fits</td>
                <td>HST/n9nk20010/n9nk20010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02020_asc.fits</td>
                <td>HST/n9nk02020/n9nk02020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12010_asn.fits</td>
                <td>HST/n8yj12010/n8yj12010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12020_mos.fits</td>
                <td>HST/n8yj12020/n8yj12020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16020_asc.fits</td>
                <td>HST/n8yj16020/n8yj16020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35020_asn.fits</td>
                <td>HST/n8yj35020/n8yj35020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23010_asn.fits</td>
                <td>HST/n8yj23010/n8yj23010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66010_mos.fits</td>
                <td>HST/n8yj66010/n8yj66010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28010_asc.fits</td>
                <td>HST/n8yj28010/n8yj28010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37010_asc.fits</td>
                <td>HST/n8yj37010/n8yj37010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26020_asn.fits</td>
                <td>HST/n9nk26020/n9nk26020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01010_asn.fits</td>
                <td>HST/n8yj01010/n8yj01010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11020_asc.fits</td>
                <td>HST/n9nk11020/n9nk11020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29020_asn.fits</td>
                <td>HST/n8yj29020/n8yj29020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08010_asc.fits</td>
                <td>HST/n8yj08010/n8yj08010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21020_asn.fits</td>
                <td>HST/n9nk21020/n9nk21020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20020_asc.fits</td>
                <td>HST/n9nk20020/n9nk20020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36010_mos.fits</td>
                <td>HST/n8yj36010/n8yj36010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21010_asc.fits</td>
                <td>HST/n9nk21010/n9nk21010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51010_mos.fits</td>
                <td>HST/n8yj51010/n8yj51010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30010_asc.fits</td>
                <td>HST/n8yj30010/n8yj30010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61010_asn.fits</td>
                <td>HST/n8yj61010/n8yj61010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03020_asc.fits</td>
                <td>HST/n9nk03020/n9nk03020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47020_asc.fits</td>
                <td>HST/n8yj47020/n8yj47020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19020_asn.fits</td>
                <td>HST/n9nk19020/n9nk19020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32010_mos.fits</td>
                <td>HST/n8yj32010/n8yj32010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38010_asn.fits</td>
                <td>HST/n8yj38010/n8yj38010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01010_asc.fits</td>
                <td>HST/n9nk01010/n9nk01010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04020_mos.fits</td>
                <td>HST/n8yj04020/n8yj04020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60010_mos.fits</td>
                <td>HST/n8yj60010/n8yj60010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33010_asn.fits</td>
                <td>HST/n8yj33010/n8yj33010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34010_asc.fits</td>
                <td>HST/n8yj34010/n8yj34010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43020_mos.fits</td>
                <td>HST/n8yj43020/n8yj43020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67010_mos.fits</td>
                <td>HST/n8yj67010/n8yj67010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04010_asn.fits</td>
                <td>HST/n9nk04010/n9nk04010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56010_mos.fits</td>
                <td>HST/n8yj56010/n8yj56010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09010_asc.fits</td>
                <td>HST/n8yj09010/n8yj09010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10020_asc.fits</td>
                <td>HST/n9nk10020/n9nk10020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09020_asn.fits</td>
                <td>HST/n9nk09020/n9nk09020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48020_mos.fits</td>
                <td>HST/n8yj48020/n8yj48020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42010_asn.fits</td>
                <td>HST/n8yj42010/n8yj42010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17010_asn.fits</td>
                <td>HST/n8yj17010/n8yj17010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09010_asn.fits</td>
                <td>HST/n9nk09010/n9nk09010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14010_asn.fits</td>
                <td>HST/n8yj14010/n8yj14010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26020_asc.fits</td>
                <td>HST/n8yj26020/n8yj26020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28020_asc.fits</td>
                <td>HST/n8yj28020/n8yj28020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35010_mos.fits</td>
                <td>HST/n8yj35010/n8yj35010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08020_asc.fits</td>
                <td>HST/n9nk08020/n9nk08020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17010_mos.fits</td>
                <td>HST/n9nk17010/n9nk17010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58010_asc.fits</td>
                <td>HST/n8yj58010/n8yj58010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66020_asn.fits</td>
                <td>HST/n8yj66020/n8yj66020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24010_asn.fits</td>
                <td>HST/n9nk24010/n9nk24010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40020_mos.fits</td>
                <td>HST/n8yj40020/n8yj40020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08020_asn.fits</td>
                <td>HST/n8yj08020/n8yj08020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01020_asc.fits</td>
                <td>HST/n9nk01020/n9nk01020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30010_asc.fits</td>
                <td>HST/n9nk30010/n9nk30010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45020_asn.fits</td>
                <td>HST/n8yj45020/n8yj45020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17020_asc.fits</td>
                <td>HST/n8yj17020/n8yj17020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23020_mos.fits</td>
                <td>HST/n9nk23020/n9nk23020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22020_asn.fits</td>
                <td>HST/n9nk22020/n9nk22020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34010_asn.fits</td>
                <td>HST/n8yj34010/n8yj34010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52010_asn.fits</td>
                <td>HST/n8yj52010/n8yj52010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36010_asn.fits</td>
                <td>HST/n8yj36010/n8yj36010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34020_asn.fits</td>
                <td>HST/n8yj34020/n8yj34020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49010_asn.fits</td>
                <td>HST/n8yj49010/n8yj49010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29010_mos.fits</td>
                <td>HST/n8yj29010/n8yj29010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26010_asc.fits</td>
                <td>HST/n9nk26010/n9nk26010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61020_mos.fits</td>
                <td>HST/n8yj61020/n8yj61020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08010_mos.fits</td>
                <td>HST/n9nk08010/n9nk08010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13020_mos.fits</td>
                <td>HST/n9nk13020/n9nk13020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21020_asc.fits</td>
                <td>HST/n8yj21020/n8yj21020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39020_mos.fits</td>
                <td>HST/n8yj39020/n8yj39020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06010_asn.fits</td>
                <td>HST/n8yj06010/n8yj06010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32020_mos.fits</td>
                <td>HST/n8yj32020/n8yj32020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07010_asn.fits</td>
                <td>HST/n8yj07010/n8yj07010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23020_asn.fits</td>
                <td>HST/n8yj23020/n8yj23020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42010_asc.fits</td>
                <td>HST/n8yj42010/n8yj42010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68020_mos.fits</td>
                <td>HST/n8yj68020/n8yj68020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53020_mos.fits</td>
                <td>HST/n8yj53020/n8yj53020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29010_asc.fits</td>
                <td>HST/n8yj29010/n8yj29010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05020_asn.fits</td>
                <td>HST/n8yj05020/n8yj05020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51010_asc.fits</td>
                <td>HST/n8yj51010/n8yj51010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12020_asc.fits</td>
                <td>HST/n9nk12020/n9nk12020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25020_mos.fits</td>
                <td>HST/n8yj25020/n8yj25020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16010_mos.fits</td>
                <td>HST/n8yj16010/n8yj16010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07010_asc.fits</td>
                <td>HST/n9nk07010/n9nk07010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19010_asc.fits</td>
                <td>HST/n9nk19010/n9nk19010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20020_mos.fits</td>
                <td>HST/n8yj20020/n8yj20020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06020_asn.fits</td>
                <td>HST/n8yj06020/n8yj06020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64020_asc.fits</td>
                <td>HST/n8yj64020/n8yj64020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29010_mos.fits</td>
                <td>HST/n9nk29010/n9nk29010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44020_asc.fits</td>
                <td>HST/n8yj44020/n8yj44020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68010_asn.fits</td>
                <td>HST/n8yj68010/n8yj68010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23010_mos.fits</td>
                <td>HST/n9nk23010/n9nk23010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55020_mos.fits</td>
                <td>HST/n8yj55020/n8yj55020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57020_asn.fits</td>
                <td>HST/n8yj57020/n8yj57020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24020_asn.fits</td>
                <td>HST/n9nk24020/n9nk24020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08010_asc.fits</td>
                <td>HST/n9nk08010/n9nk08010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22020_asc.fits</td>
                <td>HST/n8yj22020/n8yj22020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30010_asn.fits</td>
                <td>HST/n8yj30010/n8yj30010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33010_asc.fits</td>
                <td>HST/n8yj33010/n8yj33010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41020_asn.fits</td>
                <td>HST/n8yj41020/n8yj41020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59010_asn.fits</td>
                <td>HST/n8yj59010/n8yj59010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29020_mos.fits</td>
                <td>HST/n8yj29020/n8yj29020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62010_asn.fits</td>
                <td>HST/n8yj62010/n8yj62010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HLA/url/cgi-bin/fitscut.cgi?red=HST_10879_31_NIC_NIC1_F170M&amp;blue=HST_10879_31_NIC_NIC1_F110W&amp;amp;size=ALL&amp;amp;format=fits</td>
                <td>HLA/url/cgi-bin/HST_10879_31_NIC_NIC1_F170M_F110W.fits</td>
                <td>PUBLIC</td>
                <td>REMOTE</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18020_asn.fits</td>
                <td>HST/n9nk18020/n9nk18020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18020_asn.fits</td>
                <td>HST/n8yj18020/n8yj18020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26010_asc.fits</td>
                <td>HST/n8yj26010/n8yj26010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31010_mos.fits</td>
                <td>HST/n8yj31010/n8yj31010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05020_asc.fits</td>
                <td>HST/n9nk05020/n9nk05020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18010_mos.fits</td>
                <td>HST/n8yj18010/n8yj18010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19020_mos.fits</td>
                <td>HST/n8yj19020/n8yj19020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51020_mos.fits</td>
                <td>HST/n8yj51020/n8yj51020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57020_asc.fits</td>
                <td>HST/n8yj57020/n8yj57020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08010_asn.fits</td>
                <td>HST/n8yj08010/n8yj08010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19010_asn.fits</td>
                <td>HST/n8yj19010/n8yj19010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28020_mos.fits</td>
                <td>HST/n8yj28020/n8yj28020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27010_asn.fits</td>
                <td>HST/n9nk27010/n9nk27010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41020_asc.fits</td>
                <td>HST/n8yj41020/n8yj41020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01010_asc.fits</td>
                <td>HST/n8yj01010/n8yj01010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49010_asc.fits</td>
                <td>HST/n8yj49010/n8yj49010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61020_asc.fits</td>
                <td>HST/n8yj61020/n8yj61020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23020_asc.fits</td>
                <td>HST/n9nk23020/n9nk23020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23010_mos.fits</td>
                <td>HST/n8yj23010/n8yj23010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18020_mos.fits</td>
                <td>HST/n8yj18020/n8yj18020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25010_asn.fits</td>
                <td>HST/n8yj25010/n8yj25010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32020_asn.fits</td>
                <td>HST/n8yj32020/n8yj32020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64020_mos.fits</td>
                <td>HST/n8yj64020/n8yj64020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10010_asc.fits</td>
                <td>HST/n8yj10010/n8yj10010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36020_mos.fits</td>
                <td>HST/n8yj36020/n8yj36020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33020_mos.fits</td>
                <td>HST/n8yj33020/n8yj33020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01010_asn.fits</td>
                <td>HST/n9nk01010/n9nk01010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04020_asn.fits</td>
                <td>HST/n8yj04020/n8yj04020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68020_asc.fits</td>
                <td>HST/n8yj68020/n8yj68020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52020_mos.fits</td>
                <td>HST/n8yj52020/n8yj52020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26020_asn.fits</td>
                <td>HST/n8yj26020/n8yj26020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55010_asc.fits</td>
                <td>HST/n8yj55010/n8yj55010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58020_asn.fits</td>
                <td>HST/n8yj58020/n8yj58020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02010_mos.fits</td>
                <td>HST/n9nk02010/n9nk02010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33020_asn.fits</td>
                <td>HST/n8yj33020/n8yj33020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19020_asc.fits</td>
                <td>HST/n9nk19020/n9nk19020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16010_asn.fits</td>
                <td>HST/n8yj16010/n8yj16010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15010_asc.fits</td>
                <td>HST/n8yj15010/n8yj15010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34010_mos.fits</td>
                <td>HST/n8yj34010/n8yj34010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25010_asc.fits</td>
                <td>HST/n8yj25010/n8yj25010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14020_asc.fits</td>
                <td>HST/n9nk14020/n9nk14020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27010_asc.fits</td>
                <td>HST/n9nk27010/n9nk27010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15020_asc.fits</td>
                <td>HST/n8yj15020/n8yj15020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31010_asn.fits</td>
                <td>HST/n8yj31010/n8yj31010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20010_asn.fits</td>
                <td>HST/n8yj20010/n8yj20010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62020_asc.fits</td>
                <td>HST/n8yj62020/n8yj62020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37020_mos.fits</td>
                <td>HST/n8yj37020/n8yj37020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03010_asc.fits</td>
                <td>HST/n9nk03010/n9nk03010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12020_asn.fits</td>
                <td>HST/n9nk12020/n9nk12020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02020_mos.fits</td>
                <td>HST/n8yj02020/n8yj02020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19010_mos.fits</td>
                <td>HST/n9nk19010/n9nk19010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44010_asc.fits</td>
                <td>HST/n8yj44010/n8yj44010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09020_asc.fits</td>
                <td>HST/n8yj09020/n8yj09020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42010_mos.fits</td>
                <td>HST/n8yj42010/n8yj42010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62010_mos.fits</td>
                <td>HST/n8yj62010/n8yj62010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14020_asn.fits</td>
                <td>HST/n8yj14020/n8yj14020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15010_asc.fits</td>
                <td>HST/n9nk15010/n9nk15010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01020_asn.fits</td>
                <td>HST/n8yj01020/n8yj01020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52010_asc.fits</td>
                <td>HST/n8yj52010/n8yj52010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04020_mos.fits</td>
                <td>HST/n9nk04020/n9nk04020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26010_asn.fits</td>
                <td>HST/n9nk26010/n9nk26010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11020_asn.fits</td>
                <td>HST/n8yj11020/n8yj11020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47010_asc.fits</td>
                <td>HST/n8yj47010/n8yj47010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25010_asn.fits</td>
                <td>HST/n9nk25010/n9nk25010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41010_mos.fits</td>
                <td>HST/n8yj41010/n8yj41010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23010_asc.fits</td>
                <td>HST/n8yj23010/n8yj23010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53020_asn.fits</td>
                <td>HST/n8yj53020/n8yj53020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01020_asc.fits</td>
                <td>HST/n8yj01020/n8yj01020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20020_mos.fits</td>
                <td>HST/n9nk20020/n9nk20020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04010_asn.fits</td>
                <td>HST/n8yj04010/n8yj04010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07020_asn.fits</td>
                <td>HST/n8yj07020/n8yj07020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37020_asc.fits</td>
                <td>HST/n8yj37020/n8yj37020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19020_mos.fits</td>
                <td>HST/n9nk19020/n9nk19020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10020_mos.fits</td>
                <td>HST/n9nk10020/n9nk10020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17010_mos.fits</td>
                <td>HST/n8yj17010/n8yj17010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42020_asc.fits</td>
                <td>HST/n8yj42020/n8yj42020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50010_asc.fits</td>
                <td>HST/n8yj50010/n8yj50010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19010_mos.fits</td>
                <td>HST/n8yj19010/n8yj19010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58010_asn.fits</td>
                <td>HST/n8yj58010/n8yj58010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11020_asn.fits</td>
                <td>HST/n9nk11020/n9nk11020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30020_asn.fits</td>
                <td>HST/n8yj30020/n8yj30020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12010_asn.fits</td>
                <td>HST/n9nk12010/n9nk12010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25020_asc.fits</td>
                <td>HST/n9nk25020/n9nk25020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18020_asc.fits</td>
                <td>HST/n9nk18020/n9nk18020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43010_asc.fits</td>
                <td>HST/n8yj43010/n8yj43010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20010_mos.fits</td>
                <td>HST/n9nk20010/n9nk20010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35010_asc.fits</td>
                <td>HST/n8yj35010/n8yj35010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16010_asc.fits</td>
                <td>HST/n8yj16010/n8yj16010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26010_mos.fits</td>
                <td>HST/n8yj26010/n8yj26010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21010_mos.fits</td>
                <td>HST/n8yj21010/n8yj21010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47010_asn.fits</td>
                <td>HST/n8yj47010/n8yj47010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56010_asn.fits</td>
                <td>HST/n8yj56010/n8yj56010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08010_mos.fits</td>
                <td>HST/n8yj08010/n8yj08010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08020_mos.fits</td>
                <td>HST/n9nk08020/n9nk08020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37010_asn.fits</td>
                <td>HST/n8yj37010/n8yj37010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27010_mos.fits</td>
                <td>HST/n8yj27010/n8yj27010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66020_mos.fits</td>
                <td>HST/n8yj66020/n8yj66020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28020_asn.fits</td>
                <td>HST/n8yj28020/n8yj28020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02020_asn.fits</td>
                <td>HST/n8yj02020/n8yj02020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50020_asn.fits</td>
                <td>HST/n8yj50020/n8yj50020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41020_mos.fits</td>
                <td>HST/n8yj41020/n8yj41020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01010_mos.fits</td>
                <td>HST/n8yj01010/n8yj01010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk10010_asn.fits</td>
                <td>HST/n9nk10010/n9nk10010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30010_mos.fits</td>
                <td>HST/n8yj30010/n8yj30010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28020_mos.fits</td>
                <td>HST/n9nk28020/n9nk28020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17020_asn.fits</td>
                <td>HST/n9nk17020/n9nk17020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60020_asc.fits</td>
                <td>HST/n8yj60020/n8yj60020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12020_asn.fits</td>
                <td>HST/n8yj12020/n8yj12020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11010_mos.fits</td>
                <td>HST/n8yj11010/n8yj11010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50010_asn.fits</td>
                <td>HST/n8yj50010/n8yj50010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12010_asc.fits</td>
                <td>HST/n8yj12010/n8yj12010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21010_mos.fits</td>
                <td>HST/n9nk21010/n9nk21010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03020_asn.fits</td>
                <td>HST/n9nk03020/n9nk03020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51020_asc.fits</td>
                <td>HST/n8yj51020/n8yj51020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22010_asn.fits</td>
                <td>HST/n8yj22010/n8yj22010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34020_asc.fits</td>
                <td>HST/n8yj34020/n8yj34020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj01020_mos.fits</td>
                <td>HST/n8yj01020/n8yj01020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03010_mos.fits</td>
                <td>HST/n8yj03010/n8yj03010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31010_asc.fits</td>
                <td>HST/n8yj31010/n8yj31010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25020_mos.fits</td>
                <td>HST/n9nk25020/n9nk25020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62010_asc.fits</td>
                <td>HST/n8yj62010/n8yj62010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56020_mos.fits</td>
                <td>HST/n8yj56020/n8yj56020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35010_asn.fits</td>
                <td>HST/n8yj35010/n8yj35010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23020_asc.fits</td>
                <td>HST/n8yj23020/n8yj23020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24020_mos.fits</td>
                <td>HST/n9nk24020/n9nk24020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13010_mos.fits</td>
                <td>HST/n8yj13010/n8yj13010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15010_mos.fits</td>
                <td>HST/n8yj15010/n8yj15010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj29020_asc.fits</td>
                <td>HST/n8yj29020/n8yj29020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26020_mos.fits</td>
                <td>HST/n9nk26020/n9nk26020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20010_asn.fits</td>
                <td>HST/n9nk20010/n9nk20010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44020_asn.fits</td>
                <td>HST/n8yj44020/n8yj44020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16020_mos.fits</td>
                <td>HST/n8yj16020/n8yj16020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35020_mos.fits</td>
                <td>HST/n8yj35020/n8yj35020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38010_asc.fits</td>
                <td>HST/n8yj38010/n8yj38010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03010_asc.fits</td>
                <td>HST/n8yj03010/n8yj03010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63010_mos.fits</td>
                <td>HST/n8yj63010/n8yj63010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67020_mos.fits</td>
                <td>HST/n8yj67020/n8yj67020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09020_asn.fits</td>
                <td>HST/n8yj09020/n8yj09020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13020_mos.fits</td>
                <td>HST/n8yj13020/n8yj13020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24020_mos.fits</td>
                <td>HST/n8yj24020/n8yj24020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56020_asc.fits</td>
                <td>HST/n8yj56020/n8yj56020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08020_asn.fits</td>
                <td>HST/n9nk08020/n9nk08020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48020_asn.fits</td>
                <td>HST/n8yj48020/n8yj48020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22010_asc.fits</td>
                <td>HST/n8yj22010/n8yj22010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17010_asc.fits</td>
                <td>HST/n8yj17010/n8yj17010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63020_asn.fits</td>
                <td>HST/n8yj63020/n8yj63020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11020_asc.fits</td>
                <td>HST/n8yj11020/n8yj11020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18010_mos.fits</td>
                <td>HST/n9nk18010/n9nk18010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49020_asc.fits</td>
                <td>HST/n8yj49020/n8yj49020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54020_mos.fits</td>
                <td>HST/n8yj54020/n8yj54020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64010_asc.fits</td>
                <td>HST/n8yj64010/n8yj64010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14010_asc.fits</td>
                <td>HST/n9nk14010/n9nk14010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27020_asc.fits</td>
                <td>HST/n8yj27020/n8yj27020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31010_asn.fits</td>
                <td>HST/n9nk31010/n9nk31010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15010_asn.fits</td>
                <td>HST/n8yj15010/n8yj15010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17020_mos.fits</td>
                <td>HST/n8yj17020/n8yj17020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj26010_asn.fits</td>
                <td>HST/n8yj26010/n8yj26010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06020_asc.fits</td>
                <td>HST/n9nk06020/n9nk06020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12020_mos.fits</td>
                <td>HST/n9nk12020/n9nk12020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01020_asn.fits</td>
                <td>HST/n9nk01020/n9nk01020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02010_asn.fits</td>
                <td>HST/n8yj02010/n8yj02010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04020_asc.fits</td>
                <td>HST/n8yj04020/n8yj04020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04020_asc.fits</td>
                <td>HST/n9nk04020/n9nk04020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18010_asc.fits</td>
                <td>HST/n9nk18010/n9nk18010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62020_asn.fits</td>
                <td>HST/n8yj62020/n8yj62020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28020_asc.fits</td>
                <td>HST/n9nk28020/n9nk28020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27010_asn.fits</td>
                <td>HST/n8yj27010/n8yj27010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58020_mos.fits</td>
                <td>HST/n8yj58020/n8yj58020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27010_asc.fits</td>
                <td>HST/n8yj27010/n8yj27010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39010_mos.fits</td>
                <td>HST/n8yj39010/n8yj39010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13010_asc.fits</td>
                <td>HST/n8yj13010/n8yj13010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51010_asn.fits</td>
                <td>HST/n8yj51010/n8yj51010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj51020_asn.fits</td>
                <td>HST/n8yj51020/n8yj51020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60020_asn.fits</td>
                <td>HST/n8yj60020/n8yj60020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12010_asc.fits</td>
                <td>HST/n9nk12010/n9nk12010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21020_asn.fits</td>
                <td>HST/n8yj21020/n8yj21020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02010_asc.fits</td>
                <td>HST/n8yj02010/n8yj02010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45020_asc.fits</td>
                <td>HST/n8yj45020/n8yj45020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52020_asc.fits</td>
                <td>HST/n8yj52020/n8yj52020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31020_asn.fits</td>
                <td>HST/n8yj31020/n8yj31020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07020_mos.fits</td>
                <td>HST/n9nk07020/n9nk07020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26020_asc.fits</td>
                <td>HST/n9nk26020/n9nk26020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22010_asc.fits</td>
                <td>HST/n9nk22010/n9nk22010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28010_asn.fits</td>
                <td>HST/n8yj28010/n8yj28010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21010_asn.fits</td>
                <td>HST/n8yj21010/n8yj21010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06020_mos.fits</td>
                <td>HST/n9nk06020/n9nk06020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57010_asn.fits</td>
                <td>HST/n8yj57010/n8yj57010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47010_mos.fits</td>
                <td>HST/n8yj47010/n8yj47010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29010_asc.fits</td>
                <td>HST/n9nk29010/n9nk29010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07020_asc.fits</td>
                <td>HST/n8yj07020/n8yj07020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36020_asc.fits</td>
                <td>HST/n8yj36020/n8yj36020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67010_asc.fits</td>
                <td>HST/n8yj67010/n8yj67010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40010_mos.fits</td>
                <td>HST/n8yj40010/n8yj40010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24010_asc.fits</td>
                <td>HST/n8yj24010/n8yj24010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63010_asn.fits</td>
                <td>HST/n8yj63010/n8yj63010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk26010_mos.fits</td>
                <td>HST/n9nk26010/n9nk26010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33020_asc.fits</td>
                <td>HST/n8yj33020/n8yj33020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38020_asc.fits</td>
                <td>HST/n8yj38020/n8yj38020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44010_asn.fits</td>
                <td>HST/n8yj44010/n8yj44010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09010_asc.fits</td>
                <td>HST/n9nk09010/n9nk09010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj23020_mos.fits</td>
                <td>HST/n8yj23020/n8yj23020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50020_asc.fits</td>
                <td>HST/n8yj50020/n8yj50020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03020_mos.fits</td>
                <td>HST/n9nk03020/n9nk03020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07020_mos.fits</td>
                <td>HST/n8yj07020/n8yj07020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41010_asc.fits</td>
                <td>HST/n8yj41010/n8yj41010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30010_asn.fits</td>
                <td>HST/n9nk30010/n9nk30010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29020_asc.fits</td>
                <td>HST/n9nk29020/n9nk29020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj62020_mos.fits</td>
                <td>HST/n8yj62020/n8yj62020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06010_mos.fits</td>
                <td>HST/n9nk06010/n9nk06010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54010_asc.fits</td>
                <td>HST/n8yj54010/n8yj54010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15010_mos.fits</td>
                <td>HST/n9nk15010/n9nk15010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06010_mos.fits</td>
                <td>HST/n8yj06010/n8yj06010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32020_asc.fits</td>
                <td>HST/n8yj32020/n8yj32020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03010_asn.fits</td>
                <td>HST/n9nk03010/n9nk03010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09020_asc.fits</td>
                <td>HST/n9nk09020/n9nk09020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05010_mos.fits</td>
                <td>HST/n8yj05010/n8yj05010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64010_asn.fits</td>
                <td>HST/n8yj64010/n8yj64010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19010_asc.fits</td>
                <td>HST/n8yj19010/n8yj19010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24020_asc.fits</td>
                <td>HST/n8yj24020/n8yj24020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24010_mos.fits</td>
                <td>HST/n9nk24010/n9nk24010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17020_mos.fits</td>
                <td>HST/n9nk17020/n9nk17020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk08010_asn.fits</td>
                <td>HST/n9nk08010/n9nk08010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58010_mos.fits</td>
                <td>HST/n8yj58010/n8yj58010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29020_mos.fits</td>
                <td>HST/n9nk29020/n9nk29020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08020_asc.fits</td>
                <td>HST/n8yj08020/n8yj08020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67010_asn.fits</td>
                <td>HST/n8yj67010/n8yj67010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38020_mos.fits</td>
                <td>HST/n8yj38020/n8yj38020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46020_mos.fits</td>
                <td>HST/n8yj46020/n8yj46020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk20020_asn.fits</td>
                <td>HST/n9nk20020/n9nk20020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03020_asn.fits</td>
                <td>HST/n8yj03020/n8yj03020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21020_mos.fits</td>
                <td>HST/n8yj21020/n8yj21020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67020_asc.fits</td>
                <td>HST/n8yj67020/n8yj67020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39010_asc.fits</td>
                <td>HST/n8yj39010/n8yj39010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj56020_asn.fits</td>
                <td>HST/n8yj56020/n8yj56020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27020_asc.fits</td>
                <td>HST/n9nk27020/n9nk27020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66010_asc.fits</td>
                <td>HST/n8yj66010/n8yj66010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22020_mos.fits</td>
                <td>HST/n8yj22020/n8yj22020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54020_asn.fits</td>
                <td>HST/n8yj54020/n8yj54020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47020_asn.fits</td>
                <td>HST/n8yj47020/n8yj47020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43010_mos.fits</td>
                <td>HST/n8yj43010/n8yj43010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64010_mos.fits</td>
                <td>HST/n8yj64010/n8yj64010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj44020_mos.fits</td>
                <td>HST/n8yj44020/n8yj44020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40010_asn.fits</td>
                <td>HST/n8yj40010/n8yj40010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23020_asn.fits</td>
                <td>HST/n9nk23020/n9nk23020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20010_asc.fits</td>
                <td>HST/n8yj20010/n8yj20010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj08020_mos.fits</td>
                <td>HST/n8yj08020/n8yj08020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54010_asn.fits</td>
                <td>HST/n8yj54010/n8yj54010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29010_asn.fits</td>
                <td>HST/n9nk29010/n9nk29010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52020_asn.fits</td>
                <td>HST/n8yj52020/n8yj52020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38020_asn.fits</td>
                <td>HST/n8yj38020/n8yj38020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02010_asn.fits</td>
                <td>HST/n9nk02010/n9nk02010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj21010_asc.fits</td>
                <td>HST/n8yj21010/n8yj21010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40020_asn.fits</td>
                <td>HST/n8yj40020/n8yj40020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55020_asn.fits</td>
                <td>HST/n8yj55020/n8yj55020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15020_mos.fits</td>
                <td>HST/n8yj15020/n8yj15020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05010_mos.fits</td>
                <td>HST/n9nk05010/n9nk05010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk19010_asn.fits</td>
                <td>HST/n9nk19010/n9nk19010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55010_asn.fits</td>
                <td>HST/n8yj55010/n8yj55010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60020_mos.fits</td>
                <td>HST/n8yj60020/n8yj60020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12020_asc.fits</td>
                <td>HST/n8yj12020/n8yj12020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24020_asc.fits</td>
                <td>HST/n9nk24020/n9nk24020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24010_asn.fits</td>
                <td>HST/n8yj24010/n8yj24010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68010_asc.fits</td>
                <td>HST/n8yj68010/n8yj68010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20010_mos.fits</td>
                <td>HST/n8yj20010/n8yj20010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13020_asn.fits</td>
                <td>HST/n9nk13020/n9nk13020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37020_asn.fits</td>
                <td>HST/n8yj37020/n8yj37020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj52010_mos.fits</td>
                <td>HST/n8yj52010/n8yj52010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09010_mos.fits</td>
                <td>HST/n9nk09010/n9nk09010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48010_asc.fits</td>
                <td>HST/n8yj48010/n8yj48010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj67020_asn.fits</td>
                <td>HST/n8yj67020/n8yj67020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05010_asc.fits</td>
                <td>HST/n8yj05010/n8yj05010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05010_asc.fits</td>
                <td>HST/n9nk05010/n9nk05010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj12010_mos.fits</td>
                <td>HST/n8yj12010/n8yj12010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04010_mos.fits</td>
                <td>HST/n9nk04010/n9nk04010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04020_asn.fits</td>
                <td>HST/n9nk04020/n9nk04020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11010_asn.fits</td>
                <td>HST/n9nk11010/n9nk11010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj37010_mos.fits</td>
                <td>HST/n8yj37010/n8yj37010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02020_mos.fits</td>
                <td>HST/n9nk02020/n9nk02020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09010_asn.fits</td>
                <td>HST/n8yj09010/n8yj09010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46010_mos.fits</td>
                <td>HST/n8yj46010/n8yj46010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj13020_asc.fits</td>
                <td>HST/n8yj13020/n8yj13020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02010_asc.fits</td>
                <td>HST/n9nk02010/n9nk02010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22020_mos.fits</td>
                <td>HST/n9nk22020/n9nk22020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23010_asc.fits</td>
                <td>HST/n9nk23010/n9nk23010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36020_asn.fits</td>
                <td>HST/n8yj36020/n8yj36020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13010_asn.fits</td>
                <td>HST/n9nk13010/n9nk13010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj17020_asn.fits</td>
                <td>HST/n8yj17020/n8yj17020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj66020_asc.fits</td>
                <td>HST/n8yj66020/n8yj66020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53020_asc.fits</td>
                <td>HST/n8yj53020/n8yj53020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19020_asn.fits</td>
                <td>HST/n8yj19020/n8yj19020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk03010_mos.fits</td>
                <td>HST/n9nk03010/n9nk03010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46020_asc.fits</td>
                <td>HST/n8yj46020/n8yj46020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04010_mos.fits</td>
                <td>HST/n8yj04010/n8yj04010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj25020_asc.fits</td>
                <td>HST/n8yj25020/n8yj25020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02010_mos.fits</td>
                <td>HST/n8yj02010/n8yj02010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30020_mos.fits</td>
                <td>HST/n8yj30020/n8yj30020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj16020_asn.fits</td>
                <td>HST/n8yj16020/n8yj16020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05010_asn.fits</td>
                <td>HST/n9nk05010/n9nk05010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60010_asn.fits</td>
                <td>HST/n8yj60010/n8yj60010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61020_asn.fits</td>
                <td>HST/n8yj61020/n8yj61020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65020_asn.fits</td>
                <td>HST/n8yj65020/n8yj65020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15010_asn.fits</td>
                <td>HST/n9nk15010/n9nk15010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27020_asn.fits</td>
                <td>HST/n9nk27020/n9nk27020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj33010_mos.fits</td>
                <td>HST/n8yj33010/n8yj33010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28010_asc.fits</td>
                <td>HST/n9nk28010/n9nk28010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14010_mos.fits</td>
                <td>HST/n8yj14010/n8yj14010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18010_asn.fits</td>
                <td>HST/n8yj18010/n8yj18010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10020_asn.fits</td>
                <td>HST/n8yj10020/n8yj10020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj19020_asc.fits</td>
                <td>HST/n8yj19020/n8yj19020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49010_mos.fits</td>
                <td>HST/n8yj49010/n8yj49010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk29020_asn.fits</td>
                <td>HST/n9nk29020/n9nk29020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj47020_mos.fits</td>
                <td>HST/n8yj47020/n8yj47020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11010_asn.fits</td>
                <td>HST/n8yj11010/n8yj11010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj34020_mos.fits</td>
                <td>HST/n8yj34020/n8yj34020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07020_asn.fits</td>
                <td>HST/n9nk07020/n9nk07020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14010_mos.fits</td>
                <td>HST/n9nk14010/n9nk14010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59020_asc.fits</td>
                <td>HST/n8yj59020/n8yj59020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20020_asn.fits</td>
                <td>HST/n8yj20020/n8yj20020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17010_asn.fits</td>
                <td>HST/n9nk17010/n9nk17010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14010_asn.fits</td>
                <td>HST/n9nk14010/n9nk14010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06020_asn.fits</td>
                <td>HST/n9nk06020/n9nk06020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk09020_mos.fits</td>
                <td>HST/n9nk09020/n9nk09020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj64020_asn.fits</td>
                <td>HST/n8yj64020/n8yj64020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10010_mos.fits</td>
                <td>HST/n8yj10010/n8yj10010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31010_asc.fits</td>
                <td>HST/n9nk31010/n9nk31010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45010_asc.fits</td>
                <td>HST/n8yj45010/n8yj45010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj20020_asc.fits</td>
                <td>HST/n8yj20020/n8yj20020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18020_mos.fits</td>
                <td>HST/n9nk18020/n9nk18020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46020_asn.fits</td>
                <td>HST/n8yj46020/n8yj46020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06020_asc.fits</td>
                <td>HST/n8yj06020/n8yj06020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39010_asn.fits</td>
                <td>HST/n8yj39010/n8yj39010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28010_asn.fits</td>
                <td>HST/n9nk28010/n9nk28010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22010_asn.fits</td>
                <td>HST/n9nk22010/n9nk22010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59020_asn.fits</td>
                <td>HST/n8yj59020/n8yj59020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22020_asc.fits</td>
                <td>HST/n9nk22020/n9nk22020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj38010_mos.fits</td>
                <td>HST/n8yj38010/n8yj38010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17020_asc.fits</td>
                <td>HST/n9nk17020/n9nk17020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21010_asn.fits</td>
                <td>HST/n9nk21010/n9nk21010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43010_asn.fits</td>
                <td>HST/n8yj43010/n8yj43010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48010_asn.fits</td>
                <td>HST/n8yj48010/n8yj48010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj58020_asc.fits</td>
                <td>HST/n8yj58020/n8yj58020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj24020_asn.fits</td>
                <td>HST/n8yj24020/n8yj24020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57010_asc.fits</td>
                <td>HST/n8yj57010/n8yj57010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11010_mos.fits</td>
                <td>HST/n9nk11010/n9nk11010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07010_mos.fits</td>
                <td>HST/n9nk07010/n9nk07010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06010_asc.fits</td>
                <td>HST/n9nk06010/n9nk06010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30020_asc.fits</td>
                <td>HST/n9nk30020/n9nk30020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj02020_asc.fits</td>
                <td>HST/n8yj02020/n8yj02020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14020_asn.fits</td>
                <td>HST/n9nk14020/n9nk14020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05020_mos.fits</td>
                <td>HST/n9nk05020/n9nk05020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14020_asc.fits</td>
                <td>HST/n8yj14020/n8yj14020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45020_mos.fits</td>
                <td>HST/n8yj45020/n8yj45020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk18010_asn.fits</td>
                <td>HST/n9nk18010/n9nk18010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj60010_asc.fits</td>
                <td>HST/n8yj60010/n8yj60010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50010_mos.fits</td>
                <td>HST/n8yj50010/n8yj50010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25010_mos.fits</td>
                <td>HST/n9nk25010/n9nk25010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj54020_asc.fits</td>
                <td>HST/n8yj54020/n8yj54020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11020_mos.fits</td>
                <td>HST/n8yj11020/n8yj11020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11020_mos.fits</td>
                <td>HST/n9nk11020/n9nk11020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18010_asc.fits</td>
                <td>HST/n8yj18010/n8yj18010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj63020_mos.fits</td>
                <td>HST/n8yj63020/n8yj63020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk22010_mos.fits</td>
                <td>HST/n9nk22010/n9nk22010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65020_asc.fits</td>
                <td>HST/n8yj65020/n8yj65020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05010_asn.fits</td>
                <td>HST/n8yj05010/n8yj05010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40010_asc.fits</td>
                <td>HST/n8yj40010/n8yj40010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03020_asc.fits</td>
                <td>HST/n8yj03020/n8yj03020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj15020_asn.fits</td>
                <td>HST/n8yj15020/n8yj15020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21020_mos.fits</td>
                <td>HST/n9nk21020/n9nk21020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27020_asn.fits</td>
                <td>HST/n8yj27020/n8yj27020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj36010_asc.fits</td>
                <td>HST/n8yj36010/n8yj36010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk28020_asn.fits</td>
                <td>HST/n9nk28020/n9nk28020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13010_asc.fits</td>
                <td>HST/n9nk13010/n9nk13010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk30020_asn.fits</td>
                <td>HST/n9nk30020/n9nk30020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj14020_mos.fits</td>
                <td>HST/n8yj14020/n8yj14020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj28010_mos.fits</td>
                <td>HST/n8yj28010/n8yj28010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15020_mos.fits</td>
                <td>HST/n9nk15020/n9nk15020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk21020_asc.fits</td>
                <td>HST/n9nk21020/n9nk21020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25020_asn.fits</td>
                <td>HST/n9nk25020/n9nk25020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15020_asc.fits</td>
                <td>HST/n9nk15020/n9nk15020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59010_asc.fits</td>
                <td>HST/n8yj59010/n8yj59010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk24010_asc.fits</td>
                <td>HST/n9nk24010/n9nk24010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61010_mos.fits</td>
                <td>HST/n8yj61010/n8yj61010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65010_mos.fits</td>
                <td>HST/n8yj65010/n8yj65010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj27020_mos.fits</td>
                <td>HST/n8yj27020/n8yj27020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk17010_asc.fits</td>
                <td>HST/n9nk17010/n9nk17010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj09020_mos.fits</td>
                <td>HST/n8yj09020/n8yj09020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22010_mos.fits</td>
                <td>HST/n8yj22010/n8yj22010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk14020_mos.fits</td>
                <td>HST/n9nk14020/n9nk14020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk02020_asn.fits</td>
                <td>HST/n9nk02020/n9nk02020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj32010_asn.fits</td>
                <td>HST/n8yj32010/n8yj32010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk12010_mos.fits</td>
                <td>HST/n9nk12010/n9nk12010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03020_mos.fits</td>
                <td>HST/n8yj03020/n8yj03020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48020_asc.fits</td>
                <td>HST/n8yj48020/n8yj48020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07010_asc.fits</td>
                <td>HST/n8yj07010/n8yj07010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj07010_mos.fits</td>
                <td>HST/n8yj07010/n8yj07010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj11010_asc.fits</td>
                <td>HST/n8yj11010/n8yj11010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk06010_asn.fits</td>
                <td>HST/n9nk06010/n9nk06010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42020_mos.fits</td>
                <td>HST/n8yj42020/n8yj42020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk05020_asn.fits</td>
                <td>HST/n9nk05020/n9nk05020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj42020_asn.fits</td>
                <td>HST/n8yj42020/n8yj42020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65020_mos.fits</td>
                <td>HST/n8yj65020/n8yj65020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39020_asn.fits</td>
                <td>HST/n8yj39020/n8yj39020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13010_mos.fits</td>
                <td>HST/n9nk13010/n9nk13010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk15020_asn.fits</td>
                <td>HST/n9nk15020/n9nk15020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj65010_asc.fits</td>
                <td>HST/n8yj65010/n8yj65010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj68010_mos.fits</td>
                <td>HST/n8yj68010/n8yj68010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk13020_asc.fits</td>
                <td>HST/n9nk13020/n9nk13020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj04010_asc.fits</td>
                <td>HST/n8yj04010/n8yj04010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj39020_asc.fits</td>
                <td>HST/n8yj39020/n8yj39020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj61010_asc.fits</td>
                <td>HST/n8yj61010/n8yj61010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj45010_mos.fits</td>
                <td>HST/n8yj45010/n8yj45010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57010_mos.fits</td>
                <td>HST/n8yj57010/n8yj57010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49020_asn.fits</td>
                <td>HST/n8yj49020/n8yj49020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj41010_asn.fits</td>
                <td>HST/n8yj41010/n8yj41010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07010_asn.fits</td>
                <td>HST/n9nk07010/n9nk07010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj10010_asn.fits</td>
                <td>HST/n8yj10010/n8yj10010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj57020_mos.fits</td>
                <td>HST/n8yj57020/n8yj57020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj05020_mos.fits</td>
                <td>HST/n8yj05020/n8yj05020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj53010_mos.fits</td>
                <td>HST/n8yj53010/n8yj53010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk11010_asc.fits</td>
                <td>HST/n9nk11010/n9nk11010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj46010_asn.fits</td>
                <td>HST/n8yj46010/n8yj46010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj35020_asc.fits</td>
                <td>HST/n8yj35020/n8yj35020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj22020_asn.fits</td>
                <td>HST/n8yj22020/n8yj22020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk04010_asc.fits</td>
                <td>HST/n9nk04010/n9nk04010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj31020_asc.fits</td>
                <td>HST/n8yj31020/n8yj31020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01020_mos.fits</td>
                <td>HST/n9nk01020/n9nk01020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk23010_asn.fits</td>
                <td>HST/n9nk23010/n9nk23010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj06020_mos.fits</td>
                <td>HST/n8yj06020/n8yj06020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk07020_asc.fits</td>
                <td>HST/n9nk07020/n9nk07020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj49020_mos.fits</td>
                <td>HST/n8yj49020/n8yj49020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj59010_mos.fits</td>
                <td>HST/n8yj59010/n8yj59010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk27010_mos.fits</td>
                <td>HST/n9nk27010/n9nk27010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk25010_asc.fits</td>
                <td>HST/n9nk25010/n9nk25010_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj03010_asn.fits</td>
                <td>HST/n8yj03010/n8yj03010_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj48010_mos.fits</td>
                <td>HST/n8yj48010/n8yj48010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj43020_asn.fits</td>
                <td>HST/n8yj43020/n8yj43020_asn.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk01010_mos.fits</td>
                <td>HST/n9nk01010/n9nk01010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj30020_asc.fits</td>
                <td>HST/n8yj30020/n8yj30020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj18020_asc.fits</td>
                <td>HST/n8yj18020/n8yj18020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj50020_mos.fits</td>
                <td>HST/n8yj50020/n8yj50020_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj55020_asc.fits</td>
                <td>HST/n8yj55020/n8yj55020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n8yj40020_asc.fits</td>
                <td>HST/n8yj40020/n8yj40020_asc.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
            <tr>
                <td>mast:HST/product/n9nk31010_mos.fits</td>
                <td>HST/n9nk31010/n9nk31010_mos.fits</td>
                <td>PUBLIC</td>
                <td>OK</td>
                <td>anonymous</td>
            </tr>
            
        </table>
    </body>
</html>

EOT

# Download Product Files:



cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50010/n8yj50010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50010/n8yj50010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25010/n9nk25010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25010/n9nk25010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54020/n8yj54020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54020/n8yj54020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11020/n8yj11020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11020/n8yj11020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11020/n9nk11020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11020/n9nk11020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18010/n8yj18010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18010/n8yj18010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63020/n8yj63020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63020/n8yj63020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22010/n9nk22010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22010/n9nk22010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65020/n8yj65020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65020/n8yj65020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05010/n8yj05010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05010/n8yj05010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40010/n8yj40010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40010/n8yj40010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03020/n8yj03020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03020/n8yj03020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15020/n8yj15020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15020/n8yj15020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21020/n9nk21020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21020/n9nk21020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27020/n8yj27020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27020/n8yj27020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36010/n8yj36010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36010/n8yj36010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28020/n9nk28020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28020/n9nk28020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13010/n9nk13010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13010/n9nk13010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30020/n9nk30020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30020/n9nk30020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14020/n8yj14020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14020/n8yj14020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28010/n8yj28010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28010/n8yj28010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15020/n9nk15020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15020/n9nk15020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21020/n9nk21020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21020/n9nk21020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25020/n9nk25020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25020/n9nk25020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15020/n9nk15020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15020/n9nk15020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59010/n8yj59010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59010/n8yj59010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24010/n9nk24010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24010/n9nk24010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61010/n8yj61010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61010/n8yj61010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65010/n8yj65010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65010/n8yj65010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27020/n8yj27020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27020/n8yj27020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17010/n9nk17010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17010/n9nk17010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09020/n8yj09020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09020/n8yj09020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22010/n8yj22010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22010/n8yj22010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14020/n9nk14020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14020/n9nk14020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02020/n9nk02020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02020/n9nk02020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32010/n8yj32010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32010/n8yj32010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12010/n9nk12010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12010/n9nk12010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03020/n8yj03020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03020/n8yj03020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48020/n8yj48020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48020/n8yj48020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07010/n8yj07010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07010/n8yj07010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07010/n8yj07010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07010/n8yj07010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11010/n8yj11010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11010/n8yj11010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06010/n9nk06010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06010/n9nk06010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42020/n8yj42020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42020/n8yj42020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05020/n9nk05020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05020/n9nk05020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42020/n8yj42020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42020/n8yj42020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65020/n8yj65020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65020/n8yj65020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39020/n8yj39020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39020/n8yj39020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13010/n9nk13010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13010/n9nk13010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15020/n9nk15020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15020/n9nk15020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65010/n8yj65010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65010/n8yj65010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68010/n8yj68010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68010/n8yj68010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13020/n9nk13020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13020/n9nk13020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04010/n8yj04010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04010/n8yj04010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39020/n8yj39020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39020/n8yj39020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61010/n8yj61010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61010/n8yj61010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45010/n8yj45010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45010/n8yj45010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57010/n8yj57010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57010/n8yj57010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49020/n8yj49020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49020/n8yj49020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41010/n8yj41010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41010/n8yj41010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07010/n9nk07010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07010/n9nk07010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10010/n8yj10010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10010/n8yj10010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57020/n8yj57020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57020/n8yj57020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05020/n8yj05020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05020/n8yj05020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53010/n8yj53010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53010/n8yj53010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11010/n9nk11010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11010/n9nk11010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46010/n8yj46010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46010/n8yj46010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35020/n8yj35020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35020/n8yj35020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22020/n8yj22020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22020/n8yj22020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04010/n9nk04010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04010/n9nk04010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31020/n8yj31020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31020/n8yj31020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01020/n9nk01020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01020/n9nk01020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23010/n9nk23010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23010/n9nk23010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06020/n8yj06020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06020/n8yj06020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07020/n9nk07020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07020/n9nk07020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49020/n8yj49020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49020/n8yj49020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59010/n8yj59010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59010/n8yj59010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27010/n9nk27010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27010/n9nk27010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25010/n9nk25010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25010/n9nk25010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03010/n8yj03010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03010/n8yj03010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48010/n8yj48010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48010/n8yj48010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43020/n8yj43020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43020/n8yj43020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01010/n9nk01010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01010/n9nk01010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30020/n8yj30020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30020/n8yj30020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18020/n8yj18020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18020/n8yj18020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50020/n8yj50020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50020/n8yj50020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55020/n8yj55020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55020/n8yj55020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40020/n8yj40020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40020/n8yj40020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31010/n9nk31010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31010/n9nk31010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63010/n8yj63010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63010/n8yj63010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66010/n8yj66010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66010/n8yj66010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65010/n8yj65010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65010/n8yj65010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10010/n9nk10010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10010/n9nk10010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28010/n9nk28010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28010/n9nk28010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68020/n8yj68020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68020/n8yj68020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06010/n8yj06010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06010/n8yj06010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09010/n8yj09010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09010/n8yj09010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10010/n9nk10010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10010/n9nk10010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29010/n8yj29010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29010/n8yj29010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32010/n8yj32010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32010/n8yj32010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24010/n8yj24010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24010/n8yj24010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13020/n8yj13020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13020/n8yj13020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43020/n8yj43020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43020/n8yj43020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59020/n8yj59020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59020/n8yj59020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55010/n8yj55010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55010/n8yj55010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13010/n8yj13010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13010/n8yj13010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45010/n8yj45010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45010/n8yj45010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53010/n8yj53010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53010/n8yj53010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10020/n9nk10020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10020/n9nk10020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14010/n8yj14010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14010/n8yj14010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56010/n8yj56010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56010/n8yj56010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44010/n8yj44010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44010/n8yj44010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25010/n8yj25010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25010/n8yj25010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05020/n8yj05020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05020/n8yj05020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10020/n8yj10020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10020/n8yj10020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31020/n8yj31020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31020/n8yj31020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30010/n9nk30010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30010/n9nk30010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46010/n8yj46010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46010/n8yj46010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31020/n9nk31020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31020/n9nk31020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31020/n9nk31020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31020/n9nk31020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63020/n8yj63020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63020/n8yj63020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27020/n9nk27020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27020/n9nk27020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26020/n8yj26020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26020/n8yj26020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10020/n8yj10020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10020/n8yj10020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30020/n9nk30020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30020/n9nk30020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25020/n8yj25020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25020/n8yj25020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54010/n8yj54010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54010/n8yj54010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31020/n9nk31020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31020/n9nk31020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53010/n8yj53010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53010/n8yj53010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20010/n9nk20010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20010/n9nk20010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02020/n9nk02020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02020/n9nk02020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12010/n8yj12010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12010/n8yj12010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12020/n8yj12020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12020/n8yj12020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16020/n8yj16020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16020/n8yj16020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35020/n8yj35020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35020/n8yj35020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23010/n8yj23010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23010/n8yj23010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66010/n8yj66010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66010/n8yj66010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28010/n8yj28010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28010/n8yj28010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37010/n8yj37010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37010/n8yj37010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26020/n9nk26020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26020/n9nk26020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01010/n8yj01010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01010/n8yj01010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11020/n9nk11020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11020/n9nk11020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29020/n8yj29020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29020/n8yj29020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08010/n8yj08010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08010/n8yj08010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21020/n9nk21020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21020/n9nk21020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20020/n9nk20020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20020/n9nk20020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36010/n8yj36010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36010/n8yj36010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21010/n9nk21010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21010/n9nk21010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51010/n8yj51010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51010/n8yj51010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30010/n8yj30010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30010/n8yj30010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61010/n8yj61010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61010/n8yj61010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03020/n9nk03020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03020/n9nk03020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47020/n8yj47020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47020/n8yj47020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19020/n9nk19020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19020/n9nk19020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32010/n8yj32010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32010/n8yj32010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38010/n8yj38010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38010/n8yj38010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01010/n9nk01010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01010/n9nk01010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04020/n8yj04020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04020/n8yj04020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60010/n8yj60010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60010/n8yj60010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33010/n8yj33010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33010/n8yj33010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34010/n8yj34010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34010/n8yj34010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43020/n8yj43020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43020/n8yj43020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67010/n8yj67010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67010/n8yj67010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04010/n9nk04010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04010/n9nk04010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56010/n8yj56010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56010/n8yj56010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09010/n8yj09010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09010/n8yj09010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10020/n9nk10020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10020/n9nk10020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09020/n9nk09020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09020/n9nk09020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48020/n8yj48020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48020/n8yj48020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42010/n8yj42010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42010/n8yj42010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17010/n8yj17010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17010/n8yj17010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09010/n9nk09010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09010/n9nk09010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14010/n8yj14010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14010/n8yj14010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26020/n8yj26020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26020/n8yj26020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28020/n8yj28020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28020/n8yj28020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35010/n8yj35010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35010/n8yj35010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08020/n9nk08020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08020/n9nk08020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17010/n9nk17010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17010/n9nk17010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58010/n8yj58010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58010/n8yj58010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66020/n8yj66020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66020/n8yj66020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24010/n9nk24010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24010/n9nk24010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40020/n8yj40020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40020/n8yj40020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08020/n8yj08020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08020/n8yj08020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01020/n9nk01020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01020/n9nk01020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30010/n9nk30010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30010/n9nk30010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45020/n8yj45020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45020/n8yj45020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17020/n8yj17020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17020/n8yj17020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23020/n9nk23020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23020/n9nk23020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22020/n9nk22020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22020/n9nk22020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34010/n8yj34010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34010/n8yj34010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52010/n8yj52010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52010/n8yj52010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36010/n8yj36010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36010/n8yj36010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34020/n8yj34020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34020/n8yj34020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49010/n8yj49010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49010/n8yj49010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29010/n8yj29010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29010/n8yj29010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26010/n9nk26010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26010/n9nk26010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61020/n8yj61020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61020/n8yj61020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08010/n9nk08010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08010/n9nk08010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13020/n9nk13020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13020/n9nk13020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21020/n8yj21020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21020/n8yj21020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39020/n8yj39020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39020/n8yj39020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06010/n8yj06010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06010/n8yj06010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32020/n8yj32020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32020/n8yj32020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07010/n8yj07010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07010/n8yj07010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23020/n8yj23020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23020/n8yj23020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42010/n8yj42010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42010/n8yj42010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68020/n8yj68020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68020/n8yj68020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53020/n8yj53020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53020/n8yj53020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29010/n8yj29010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29010/n8yj29010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05020/n8yj05020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05020/n8yj05020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51010/n8yj51010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51010/n8yj51010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12020/n9nk12020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12020/n9nk12020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25020/n8yj25020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25020/n8yj25020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16010/n8yj16010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16010/n8yj16010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07010/n9nk07010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07010/n9nk07010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19010/n9nk19010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19010/n9nk19010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20020/n8yj20020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20020/n8yj20020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06020/n8yj06020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06020/n8yj06020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64020/n8yj64020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64020/n8yj64020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29010/n9nk29010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29010/n9nk29010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44020/n8yj44020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44020/n8yj44020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68010/n8yj68010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68010/n8yj68010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23010/n9nk23010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23010/n9nk23010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55020/n8yj55020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55020/n8yj55020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57020/n8yj57020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57020/n8yj57020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24020/n9nk24020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24020/n9nk24020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08010/n9nk08010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08010/n9nk08010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22020/n8yj22020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22020/n8yj22020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30010/n8yj30010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30010/n8yj30010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33010/n8yj33010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33010/n8yj33010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41020/n8yj41020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41020/n8yj41020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59010/n8yj59010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59010/n8yj59010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29020/n8yj29020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29020/n8yj29020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62010/n8yj62010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62010/n8yj62010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HLA/url/cgi-bin/fitscut.cgi?red=HST_10879_31_NIC_NIC1_F170M&blue=HST_10879_31_NIC_NIC1_F110W&amp;size=ALL&amp;format=fits
                  To: ${DOWNLOAD_FOLDER}/HLA/url/cgi-bin/HST_10879_31_NIC_NIC1_F170M_F110W.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HLA/url/cgi-bin/HST_10879_31_NIC_NIC1_F170M_F110W.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HLA/url/cgi-bin/fitscut.cgi%3Fred%3DHST_10879_31_NIC_NIC1_F170M%26blue%3DHST_10879_31_NIC_NIC1_F110W%26amp%3Bsize%3DALL%26amp%3Bformat%3Dfits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18020/n9nk18020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18020/n9nk18020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18020/n8yj18020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18020/n8yj18020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26010/n8yj26010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26010/n8yj26010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31010/n8yj31010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31010/n8yj31010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05020/n9nk05020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05020/n9nk05020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18010/n8yj18010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18010/n8yj18010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19020/n8yj19020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19020/n8yj19020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51020/n8yj51020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51020/n8yj51020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57020/n8yj57020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57020/n8yj57020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08010/n8yj08010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08010/n8yj08010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19010/n8yj19010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19010/n8yj19010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28020/n8yj28020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28020/n8yj28020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27010/n9nk27010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27010/n9nk27010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41020/n8yj41020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41020/n8yj41020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01010/n8yj01010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01010/n8yj01010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49010/n8yj49010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49010/n8yj49010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61020/n8yj61020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61020/n8yj61020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23020/n9nk23020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23020/n9nk23020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23010/n8yj23010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23010/n8yj23010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18020/n8yj18020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18020/n8yj18020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25010/n8yj25010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25010/n8yj25010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32020/n8yj32020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32020/n8yj32020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64020/n8yj64020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64020/n8yj64020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10010/n8yj10010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10010/n8yj10010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36020/n8yj36020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36020/n8yj36020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33020/n8yj33020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33020/n8yj33020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01010/n9nk01010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01010/n9nk01010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04020/n8yj04020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04020/n8yj04020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68020/n8yj68020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68020/n8yj68020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52020/n8yj52020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52020/n8yj52020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26020/n8yj26020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26020/n8yj26020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55010/n8yj55010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55010/n8yj55010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58020/n8yj58020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58020/n8yj58020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02010/n9nk02010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02010/n9nk02010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33020/n8yj33020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33020/n8yj33020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19020/n9nk19020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19020/n9nk19020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16010/n8yj16010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16010/n8yj16010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15010/n8yj15010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15010/n8yj15010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34010/n8yj34010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34010/n8yj34010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25010/n8yj25010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25010/n8yj25010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14020/n9nk14020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14020/n9nk14020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27010/n9nk27010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27010/n9nk27010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15020/n8yj15020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15020/n8yj15020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31010/n8yj31010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31010/n8yj31010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20010/n8yj20010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20010/n8yj20010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62020/n8yj62020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62020/n8yj62020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37020/n8yj37020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37020/n8yj37020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03010/n9nk03010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03010/n9nk03010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12020/n9nk12020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12020/n9nk12020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02020/n8yj02020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02020/n8yj02020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19010/n9nk19010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19010/n9nk19010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44010/n8yj44010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44010/n8yj44010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09020/n8yj09020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09020/n8yj09020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42010/n8yj42010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42010/n8yj42010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62010/n8yj62010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62010/n8yj62010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14020/n8yj14020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14020/n8yj14020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15010/n9nk15010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15010/n9nk15010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01020/n8yj01020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01020/n8yj01020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52010/n8yj52010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52010/n8yj52010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04020/n9nk04020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04020/n9nk04020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26010/n9nk26010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26010/n9nk26010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11020/n8yj11020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11020/n8yj11020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47010/n8yj47010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47010/n8yj47010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25010/n9nk25010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25010/n9nk25010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41010/n8yj41010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41010/n8yj41010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23010/n8yj23010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23010/n8yj23010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53020/n8yj53020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53020/n8yj53020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01020/n8yj01020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01020/n8yj01020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20020/n9nk20020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20020/n9nk20020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04010/n8yj04010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04010/n8yj04010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07020/n8yj07020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07020/n8yj07020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37020/n8yj37020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37020/n8yj37020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19020/n9nk19020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19020/n9nk19020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10020/n9nk10020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10020/n9nk10020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17010/n8yj17010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17010/n8yj17010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj42020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj42020/n8yj42020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj42020/n8yj42020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj42020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50010/n8yj50010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50010/n8yj50010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19010/n8yj19010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19010/n8yj19010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58010/n8yj58010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58010/n8yj58010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11020/n9nk11020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11020/n9nk11020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30020/n8yj30020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30020/n8yj30020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12010/n9nk12010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12010/n9nk12010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25020/n9nk25020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25020/n9nk25020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18020/n9nk18020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18020/n9nk18020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43010/n8yj43010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43010/n8yj43010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20010/n9nk20010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20010/n9nk20010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35010/n8yj35010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35010/n8yj35010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16010/n8yj16010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16010/n8yj16010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26010/n8yj26010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26010/n8yj26010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21010/n8yj21010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21010/n8yj21010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47010/n8yj47010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47010/n8yj47010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56010/n8yj56010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56010/n8yj56010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08010/n8yj08010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08010/n8yj08010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08020/n9nk08020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08020/n9nk08020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37010/n8yj37010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37010/n8yj37010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27010/n8yj27010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27010/n8yj27010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66020/n8yj66020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66020/n8yj66020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28020/n8yj28020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28020/n8yj28020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02020/n8yj02020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02020/n8yj02020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50020/n8yj50020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50020/n8yj50020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41020/n8yj41020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41020/n8yj41020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01010/n8yj01010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01010/n8yj01010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk10010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk10010/n9nk10010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk10010/n9nk10010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk10010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30010/n8yj30010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30010/n8yj30010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28020/n9nk28020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28020/n9nk28020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17020/n9nk17020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17020/n9nk17020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60020/n8yj60020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60020/n8yj60020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12020/n8yj12020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12020/n8yj12020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11010/n8yj11010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11010/n8yj11010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50010/n8yj50010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50010/n8yj50010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12010/n8yj12010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12010/n8yj12010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21010/n9nk21010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21010/n9nk21010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03020/n9nk03020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03020/n9nk03020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51020/n8yj51020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51020/n8yj51020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22010/n8yj22010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22010/n8yj22010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34020/n8yj34020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34020/n8yj34020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj01020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj01020/n8yj01020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj01020/n8yj01020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj01020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03010/n8yj03010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03010/n8yj03010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31010/n8yj31010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31010/n8yj31010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk25020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk25020/n9nk25020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk25020/n9nk25020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk25020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62010/n8yj62010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62010/n8yj62010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56020/n8yj56020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56020/n8yj56020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35010/n8yj35010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35010/n8yj35010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23020/n8yj23020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23020/n8yj23020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24020/n9nk24020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24020/n9nk24020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13010/n8yj13010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13010/n8yj13010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15010/n8yj15010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15010/n8yj15010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj29020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj29020/n8yj29020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj29020/n8yj29020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj29020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26020/n9nk26020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26020/n9nk26020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20010/n9nk20010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20010/n9nk20010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44020/n8yj44020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44020/n8yj44020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16020/n8yj16020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16020/n8yj16020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj35020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj35020/n8yj35020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj35020/n8yj35020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj35020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38010/n8yj38010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38010/n8yj38010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03010/n8yj03010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03010/n8yj03010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63010/n8yj63010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63010/n8yj63010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67020/n8yj67020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67020/n8yj67020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09020/n8yj09020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09020/n8yj09020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13020/n8yj13020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13020/n8yj13020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24020/n8yj24020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24020/n8yj24020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56020/n8yj56020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56020/n8yj56020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08020/n9nk08020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08020/n9nk08020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48020/n8yj48020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48020/n8yj48020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22010/n8yj22010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22010/n8yj22010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17010/n8yj17010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17010/n8yj17010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63020/n8yj63020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63020/n8yj63020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11020/n8yj11020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11020/n8yj11020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18010/n9nk18010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18010/n9nk18010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49020/n8yj49020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49020/n8yj49020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54020/n8yj54020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54020/n8yj54020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64010/n8yj64010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64010/n8yj64010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14010/n9nk14010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14010/n9nk14010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27020/n8yj27020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27020/n8yj27020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31010/n9nk31010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31010/n9nk31010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15010/n8yj15010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15010/n8yj15010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17020/n8yj17020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17020/n8yj17020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj26010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj26010/n8yj26010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj26010/n8yj26010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj26010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06020/n9nk06020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06020/n9nk06020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12020/n9nk12020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12020/n9nk12020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk01020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk01020/n9nk01020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk01020/n9nk01020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk01020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02010/n8yj02010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02010/n8yj02010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04020/n8yj04020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04020/n8yj04020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04020/n9nk04020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04020/n9nk04020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18010/n9nk18010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18010/n9nk18010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62020/n8yj62020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62020/n8yj62020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28020/n9nk28020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28020/n9nk28020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27010/n8yj27010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27010/n8yj27010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58020/n8yj58020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58020/n8yj58020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj27010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj27010/n8yj27010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj27010/n8yj27010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj27010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39010/n8yj39010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39010/n8yj39010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13010/n8yj13010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13010/n8yj13010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51010/n8yj51010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51010/n8yj51010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj51020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj51020/n8yj51020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj51020/n8yj51020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj51020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60020/n8yj60020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60020/n8yj60020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk12010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk12010/n9nk12010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk12010/n9nk12010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk12010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21020/n8yj21020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21020/n8yj21020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02010/n8yj02010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02010/n8yj02010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45020/n8yj45020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45020/n8yj45020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52020/n8yj52020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52020/n8yj52020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj31020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj31020/n8yj31020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj31020/n8yj31020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj31020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07020/n9nk07020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07020/n9nk07020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26020/n9nk26020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26020/n9nk26020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22010/n9nk22010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22010/n9nk22010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj28010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj28010/n8yj28010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj28010/n8yj28010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj28010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21010/n8yj21010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21010/n8yj21010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06020/n9nk06020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06020/n9nk06020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57010/n8yj57010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57010/n8yj57010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47010/n8yj47010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47010/n8yj47010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29010/n9nk29010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29010/n9nk29010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07020/n8yj07020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07020/n8yj07020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36020/n8yj36020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36020/n8yj36020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67010/n8yj67010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67010/n8yj67010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40010/n8yj40010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40010/n8yj40010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24010/n8yj24010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24010/n8yj24010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj63010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj63010/n8yj63010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj63010/n8yj63010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj63010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk26010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk26010/n9nk26010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk26010/n9nk26010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk26010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33020/n8yj33020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33020/n8yj33020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38020/n8yj38020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38020/n8yj38020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44010/n8yj44010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44010/n8yj44010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09010/n9nk09010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09010/n9nk09010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj23020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj23020/n8yj23020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj23020/n8yj23020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj23020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj50020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj50020/n8yj50020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj50020/n8yj50020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj50020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03020/n9nk03020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03020/n9nk03020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj07020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj07020/n8yj07020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj07020/n8yj07020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj07020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj41010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj41010/n8yj41010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj41010/n8yj41010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj41010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30010/n9nk30010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30010/n9nk30010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29020/n9nk29020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29020/n9nk29020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj62020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj62020/n8yj62020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj62020/n8yj62020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj62020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06010/n9nk06010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06010/n9nk06010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54010/n8yj54010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54010/n8yj54010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15010/n9nk15010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15010/n9nk15010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06010/n8yj06010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06010/n8yj06010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj32020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj32020/n8yj32020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj32020/n8yj32020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj32020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03010/n9nk03010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03010/n9nk03010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09020/n9nk09020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09020/n9nk09020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05010/n8yj05010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05010/n8yj05010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64010/n8yj64010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64010/n8yj64010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19010/n8yj19010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19010/n8yj19010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24020/n8yj24020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24020/n8yj24020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24010/n9nk24010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24010/n9nk24010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17020/n9nk17020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17020/n9nk17020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk08010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk08010/n9nk08010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk08010/n9nk08010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk08010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58010/n8yj58010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58010/n8yj58010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29020/n9nk29020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29020/n9nk29020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08020/n8yj08020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08020/n8yj08020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67010/n8yj67010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67010/n8yj67010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38020/n8yj38020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38020/n8yj38020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46020/n8yj46020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46020/n8yj46020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk20020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk20020/n9nk20020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk20020/n9nk20020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk20020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj03020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj03020/n8yj03020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj03020/n8yj03020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj03020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21020/n8yj21020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21020/n8yj21020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67020/n8yj67020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67020/n8yj67020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39010/n8yj39010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39010/n8yj39010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj56020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj56020/n8yj56020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj56020/n8yj56020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj56020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27020/n9nk27020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27020/n9nk27020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66010/n8yj66010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66010/n8yj66010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj22020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj22020/n8yj22020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj22020/n8yj22020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj22020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54020/n8yj54020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54020/n8yj54020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47020/n8yj47020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47020/n8yj47020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43010/n8yj43010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43010/n8yj43010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64010/n8yj64010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64010/n8yj64010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj44020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj44020/n8yj44020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj44020/n8yj44020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj44020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40010/n8yj40010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40010/n8yj40010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23020/n9nk23020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23020/n9nk23020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20010/n8yj20010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20010/n8yj20010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj08020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj08020/n8yj08020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj08020/n8yj08020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj08020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj54010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj54010/n8yj54010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj54010/n8yj54010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj54010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29010/n9nk29010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29010/n9nk29010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52020/n8yj52020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52020/n8yj52020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38020/n8yj38020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38020/n8yj38020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02010/n9nk02010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02010/n9nk02010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj21010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj21010/n8yj21010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj21010/n8yj21010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj21010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj40020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj40020/n8yj40020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj40020/n8yj40020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj40020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55020/n8yj55020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55020/n8yj55020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj15020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj15020/n8yj15020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj15020/n8yj15020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj15020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05010/n9nk05010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05010/n9nk05010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk19010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk19010/n9nk19010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk19010/n9nk19010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk19010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj55010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj55010/n8yj55010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj55010/n8yj55010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj55010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60020/n8yj60020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60020/n8yj60020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12020/n8yj12020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12020/n8yj12020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk24020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk24020/n9nk24020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk24020/n9nk24020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk24020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24010/n8yj24010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24010/n8yj24010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj68010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj68010/n8yj68010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj68010/n8yj68010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj68010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20010/n8yj20010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20010/n8yj20010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13020/n9nk13020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13020/n9nk13020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37020/n8yj37020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37020/n8yj37020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj52010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj52010/n8yj52010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj52010/n8yj52010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj52010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09010/n9nk09010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09010/n9nk09010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48010/n8yj48010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48010/n8yj48010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj67020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj67020/n8yj67020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj67020/n8yj67020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj67020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj05010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj05010/n8yj05010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj05010/n8yj05010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj05010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05010/n9nk05010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05010/n9nk05010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj12010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj12010/n8yj12010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj12010/n8yj12010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj12010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04010/n9nk04010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04010/n9nk04010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk04020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk04020/n9nk04020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk04020/n9nk04020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk04020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11010/n9nk11010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11010/n9nk11010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj37010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj37010/n8yj37010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj37010/n8yj37010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj37010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02020/n9nk02020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02020/n9nk02020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj09010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj09010/n8yj09010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj09010/n8yj09010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj09010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46010/n8yj46010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46010/n8yj46010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj13020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj13020/n8yj13020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj13020/n8yj13020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj13020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk02010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk02010/n9nk02010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk02010/n9nk02010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk02010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22020/n9nk22020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22020/n9nk22020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk23010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk23010/n9nk23010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk23010/n9nk23010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk23010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj36020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj36020/n8yj36020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj36020/n8yj36020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj36020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk13010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk13010/n9nk13010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk13010/n9nk13010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk13010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj17020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj17020/n8yj17020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj17020/n8yj17020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj17020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj66020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj66020/n8yj66020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj66020/n8yj66020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj66020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj53020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj53020/n8yj53020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj53020/n8yj53020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj53020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19020/n8yj19020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19020/n8yj19020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk03010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk03010/n9nk03010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk03010/n9nk03010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk03010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46020/n8yj46020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46020/n8yj46020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj04010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj04010/n8yj04010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj04010/n8yj04010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj04010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj25020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj25020/n8yj25020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj25020/n8yj25020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj25020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02010/n8yj02010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02010/n8yj02010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj30020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj30020/n8yj30020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj30020/n8yj30020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj30020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj16020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj16020/n8yj16020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj16020/n8yj16020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj16020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05010/n9nk05010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05010/n9nk05010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60010/n8yj60010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60010/n8yj60010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj61020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj61020/n8yj61020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj61020/n8yj61020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj61020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj65020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj65020/n8yj65020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj65020/n8yj65020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj65020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk15010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk15010/n9nk15010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk15010/n9nk15010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk15010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk27020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk27020/n9nk27020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk27020/n9nk27020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk27020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj33010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj33010/n8yj33010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj33010/n8yj33010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj33010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28010/n9nk28010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28010/n9nk28010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14010/n8yj14010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14010/n8yj14010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj18010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj18010/n8yj18010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj18010/n8yj18010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj18010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10020/n8yj10020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10020/n8yj10020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj19020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj19020/n8yj19020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj19020/n8yj19020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj19020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj49010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj49010/n8yj49010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj49010/n8yj49010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj49010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk29020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk29020/n9nk29020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk29020/n9nk29020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk29020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj47020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj47020/n8yj47020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj47020/n8yj47020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj47020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj11010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj11010/n8yj11010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj11010/n8yj11010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj11010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj34020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj34020/n8yj34020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj34020/n8yj34020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj34020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07020/n9nk07020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07020/n9nk07020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14010/n9nk14010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14010/n9nk14010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59020/n8yj59020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59020/n8yj59020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20020/n8yj20020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20020/n8yj20020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17010/n9nk17010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17010/n9nk17010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14010/n9nk14010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14010/n9nk14010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06020/n9nk06020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06020/n9nk06020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk09020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk09020/n9nk09020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk09020/n9nk09020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk09020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj64020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj64020/n8yj64020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj64020/n8yj64020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj64020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj10010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj10010/n8yj10010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj10010/n8yj10010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj10010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk31010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk31010/n9nk31010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk31010/n9nk31010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk31010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45010/n8yj45010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45010/n8yj45010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj20020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj20020/n8yj20020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj20020/n8yj20020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj20020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18020/n9nk18020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18020/n9nk18020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj46020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj46020/n8yj46020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj46020/n8yj46020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj46020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj06020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj06020/n8yj06020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj06020/n8yj06020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj06020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj39010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj39010/n8yj39010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj39010/n8yj39010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj39010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk28010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk28010/n9nk28010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk28010/n9nk28010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk28010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22010/n9nk22010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22010/n9nk22010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj59020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj59020/n8yj59020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj59020/n8yj59020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj59020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk22020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk22020/n9nk22020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk22020/n9nk22020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk22020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj38010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj38010/n8yj38010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj38010/n8yj38010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj38010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk17020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk17020/n9nk17020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk17020/n9nk17020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk17020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk21010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk21010/n9nk21010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk21010/n9nk21010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk21010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj43010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj43010/n8yj43010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj43010/n8yj43010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj43010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj48010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj48010/n8yj48010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj48010/n8yj48010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj48010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj58020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj58020/n8yj58020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj58020/n8yj58020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj58020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj24020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj24020/n8yj24020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj24020/n8yj24020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj24020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj57010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj57010/n8yj57010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj57010/n8yj57010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj57010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk11010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk11010/n9nk11010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk11010/n9nk11010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk11010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk07010_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk07010/n9nk07010_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk07010/n9nk07010_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk07010_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk06010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk06010/n9nk06010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk06010/n9nk06010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk06010_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk30020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk30020/n9nk30020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk30020/n9nk30020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk30020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj02020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj02020/n8yj02020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj02020/n8yj02020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj02020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk14020_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk14020/n9nk14020_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk14020/n9nk14020_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk14020_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk05020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk05020/n9nk05020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk05020/n9nk05020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk05020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj14020_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj14020/n8yj14020_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj14020/n8yj14020_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj14020_asc.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj45020_mos.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj45020/n8yj45020_mos.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj45020/n8yj45020_mos.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj45020_mos.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n9nk18010_asn.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n9nk18010/n9nk18010_asn.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n9nk18010/n9nk18010_asn.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n9nk18010_asn.fits'





cat <<EOT
<<< Downloading File: mast:HST/product/n8yj60010_asc.fits
                  To: ${DOWNLOAD_FOLDER}/HST/n8yj60010/n8yj60010_asc.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output ./${DOWNLOAD_FOLDER}'/HST/n8yj60010/n8yj60010_asc.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2024-07-03T0023.sh&uri=mast:HST/product/n8yj60010_asc.fits'




exit 0
