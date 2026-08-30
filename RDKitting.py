import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


class RDKitTools:
    def __init__(self,name,SMILES):
        self.molecule = Chem.MolFromSmiles(SMILES)
        self.name = name

    def draw_mol(self):
        img = Draw.MolToImage(self.molecule,size = (800,800))

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(self.name)
        plt.show()

    def get_properties(self,additional_props=None):
        self.additional_props = additional_props

        mass = Descriptors.MolWt(self.molecule)
        hba = Descriptors.NumHAcceptors(self.molecule)
        hbd = Descriptors.NumHDonors(self.molecule)
        logp = Descriptors.MolLogP(self.molecule)
        rot_bonds = Descriptors.NumRotatableBonds(self.molecule)
        print(f'Quick description of {self.name}')
        print('=' * 50)
        print(f'Computed mass: {mass:.2f}')
        print(f'Computed hydrogen bond acceptor(s): {hba}')
        print(f'Computed hydrogen bond donor(s): {hbd}')
        print(f'Computed rotatable bond(s): {rot_bonds}')
        print(f'Computed LogP: {logp:.2f}')
        if mass > 500 or logp > 5 or hbd > 5 or hba > 10:
            print('Predicted poor oral bioavailability')
        else:
            print('Predicted good oral bioavailability')

        print('=' * 50)

        if self.additional_props is not None:
            for prop in self.additional_props:
                property = Descriptors.CalcMolDescriptors(self.molecule)
                print(f'Computed {prop}: {property[prop]:.4f}')
            print('=' * 50)

    def get_similarity(self, compared_molecule,threshold=0.75):
        self.target_mol_fp = Chem.RDKFingerprint(self.molecule)

        self.all_similarity = []
        self.acceptable_similarity = []

        for molecules in compared_molecule:
            if isinstance(compared_molecule, dict):
                comp_mol = Chem.MolFromSmiles(compared_molecule[molecules])
                compared_mol_fp = Chem.RDKFingerprint(comp_mol)
                similarity = DataStructs.FingerprintSimilarity(self.target_mol_fp, compared_mol_fp)
                self.all_similarity.append([molecules,similarity])
                if similarity >= threshold:
                    print(f"Tanimoto Similarity Index between {self.name} and {molecules}: {similarity:.2f}")
                    self.acceptable_similarity.append([molecules,similarity])
            else:
                comp_mol = Chem.MolFromSmiles(molecules)
                compared_mol_fp = Chem.RDKFingerprint(comp_mol)
                similarity = DataStructs.FingerprintSimilarity(self.target_mol_fp, compared_mol_fp)
                self.all_similarity.append([molecules,similarity])
                if similarity >= threshold:
                    print(f"Tanimoto Similarity Index between {self.name} and {molecules}: {similarity:.2f}")
                    self.acceptable_similarity.append([molecules,similarity])
        print('=' * 50)

        self.acceptable_similarity = pd.DataFrame(self.acceptable_similarity,columns=['Molecule','Similarity'])
        self.all_similarity = pd.DataFrame(self.all_similarity,columns=['Molecule','Similarity'])
        print('Acceptable similarity:')
        print(self.acceptable_similarity)
        print('=' * 50)
        print('All Similarity:')
        print(self.all_similarity)

        print('=' * 50)
      

    def search_tool(self,database,criteria_desc=None,criteria_desc_val=None, criteria_similarity=0):
        self.target_mol_fp = Chem.RDKFingerprint(self.molecule)
        self.database = database
        self.criteria_desc = criteria_desc
        self.criteria_similarity = criteria_similarity
        # search through database to find satisfactory molecules
        # can be based on fp, descriptors ...

        mol_df = pd.read_csv(self.database)
        smiles_to_search = mol_df['SMILES']

        # first get similarity match
        matched_similarity = []
        for molecule in smiles_to_search:
            comp_mol = Chem.MolFromSmiles(molecule)
            compared_mol_fp = Chem.RDKFingerprint(comp_mol)
            similarity = DataStructs.FingerprintSimilarity(self.target_mol_fp, compared_mol_fp)
            if similarity >= criteria_similarity:
                matched_similarity.append([molecule,similarity])

        matched_similarity_df = pd.DataFrame(matched_similarity,columns=['Molecule','Similarity'])
        print(matched_similarity_df)
        print('=' * 50)

        # now compare descriptors ...
        criteria_met_list = []
        full_prop_list = []
        if criteria_desc is not None:

            for molecule in matched_similarity_df['Molecule']:
                comp_mol = Chem.MolFromSmiles(molecule)
                comp_mol_props = Descriptors.CalcMolDescriptors(comp_mol)
                # the asterisk helps unpack the dict keys/vals out of a list :)
                full_prop_list.append([molecule,*comp_mol_props.values()])

            full_desc_df = pd.DataFrame(full_prop_list, columns=['SMILES', *comp_mol_props.keys()])

            selected_desc_df = full_desc_df[criteria_desc]
            selected_desc_df.to_csv('Full_Criteria_Data_Desc.csv', index=False)
            print(selected_desc_df)
            criteria_desc.pop(0)
            # to remove SMILES from list

            print('=' * 50)
            print(f'PLEASE NOTE:\nTHE FILTERING VIA CRITERIA DESC AND VAL ASSUMES THAT YOU WANT TO FILTER IN THE ORDER GIVEN AND ALSO THAT YOU WISH TO FIND ALL VALUES THAT ARE >= THAN THE VAL INPUTTED!\nIF THIS IS NOT INTENDED, ONLY LOOK AT THE FIRST FILTERED DF AND SELECT MANUALLY!')
            print('='*50)

            count = 0
            for criteria in criteria_desc:
                print(f'Filtering for criterion: {criteria} at {criteria_desc_val[count]}...')
                filtered_data = selected_desc_df[selected_desc_df[criteria] >= criteria_desc_val[count]]
                count += 1
                selected_desc_df = filtered_data
                print(filtered_data)
                print('=' * 50)

            if filtered_data is not None:
                print('Filtered Data:')
                print(filtered_data)
                filtered_data.to_csv('Filtered_Data_Desc.csv', index=False)

            print('=' * 50)


smile = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
name = 'Caffeine'

additional_props = ['MaxAbsPartialCharge', 'MinPartialCharge','NumValenceElectrons']
mols_to_compare = {'Theobromine':'Cn1cnc2c1c(=O)[nH]c(=O)n2C','Theophylline': 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'}
#mols_to_compare = ['Cn1cnc2c1c(=O)[nH]c(=O)n2C','CN1C2=C(C(=O)N(C1=O)C)NC=N2'] # CAN ALSO JUST BE A SMILES LIST
threshold = 0.9

# SMILES MUST BE LEFT IN THE DESC !!!
criteria_desc = ['SMILES','MolWt','MolLogP']
criteria_similarity = 0.75
database = 'smiles_list.csv'
criteria_desc_val = [200, -0.56]
# TOOL ASSUMES >= FOR ALL DESC, IT ALSO WORKS CHRONOLOGICALLY !!!

test = RDKitTools(name, smile)
test.draw_mol()
test.get_properties(additional_props)
test.get_similarity(mols_to_compare,threshold=threshold)
test.search_tool(database,criteria_desc, criteria_desc_val,criteria_similarity)
