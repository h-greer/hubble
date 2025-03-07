curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01050_asc.fits" --output "MAST_2025-03-02T22_29_29.259Z/HST/n8ku01050_asc.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01050_jif.fits" --output "MAST_2025-03-02T22_29_29.259Z/HST/n8ku01050_jif.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01ffq_cal.fits" --output "MAST_2025-03-02T22_29_29.259Z/HST/n8ku01ffq_cal.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=N8KU01050%2Fn8ku01fgq_cal.fits" --output "MAST_2025-03-02T22_29_29.259Z/HST/n8ku01fgq_cal.fits" --fail --create-dirs

curl -H "Authorization: token $MAST_API_TOKEN" -L -X POST "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product_zip" -F "mission=HST" -F "structure=flat" -F uri="N8KU01050/n8ku01050_asc.fits,N8KU01050/n8ku01050_jif.fits,N8KU01050/n8ku01ffq_cal.fits,N8KU01050/n8ku01fgq_cal.fits" -F "manifestonly=true" --output "MAST_2025-03-02T22_29_29.259Z/MANIFEST.html" --fail --create-dirs

