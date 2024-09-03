.PHONY: clean

clean:
	# Supprimer le répertoire __pycache__ dans ./aci
	rm -rf ./aci/__pycache__
	# Supprimer le répertoire ./data/data0 s'il existe
	rm -rf ./data/data0
	# Supprimer les répertoires ./data/sealevel_data* s'ils existent
	rm -rf ./data/sealevel_data*
	# Supprimer les fichiers avec un tilde (~) dans ./aci et ./tests
	find ./aci -type f -name '*~' -exec rm -f {} +
	find ./tests -type f -name '*~' -exec rm -f {} +
	# Afficher un message de succès
	@echo -e "\033[32mNettoyage terminé avec succès\033[0m"

test:
	# Exécuter tous les tests dans ./tests
	cd tests && PYTHONWARNINGS="ignore::DeprecationWarning,ignore::PendingDeprecationWarning,ignore::FutureWarning" python -m unittest discover -s . -v

coverage:
	# Exécuter tous les tests dans ./tests
	cd tests && PYTHONWARNINGS="ignore::DeprecationWarning,ignore::PendingDeprecationWarning,ignore::FutureWarning" python -m coverage run -m unittest && python -m coverage report 
