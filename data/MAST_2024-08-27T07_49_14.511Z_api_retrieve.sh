curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01050_asn.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/n8ku01050_asn.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01050_mos.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/n8ku01050_mos.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01050_asc.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/n8ku01050_asc.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01ffq_cal.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/n8ku01ffq_cal.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01fgq_cal.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/n8ku01fgq_cal.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fsbj1738kn_flt.fits" --output "MAST_2024-08-27T07_49_07.684Z/HST/sbj1738kn_flt.fits" --fail --create-dirs

curl -H "Authorization: token $MAST_API_TOKEN" -L -X POST "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product_zip" -F "mission=HST" -F "structure=flat" -F uri="N8KU01050/n8ku01050_asn.fits,N8KU01050/n8ku01050_mos.fits,N8KU01050/n8ku01050_asc.fits,N8KU01050/n8ku01ffq_cal.fits,N8KU01050/n8ku01fgq_cal.fits,N8KU01050/sbj1738kn_flt.fits" -F "manifestonly=true" --output "MAST_2024-08-27T07_49_07.684Z/MANIFEST.html" --fail --create-dirs

