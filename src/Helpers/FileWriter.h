/*
 * FileWriter.h
 *
 *  Created on: Feb 22, 2019
 *      Author: basti
 */

#ifndef FILEWRITER_H_
#define FILEWRITER_H_

#include <string.h>
#include <map>

typedef unsigned long long t_ull;

class FileWriter {
public:
	FileWriter();
	~FileWriter();

	// Manage Output File
	bool FileOpen();
	void FileClose();

	bool FileInit(std::string &sFileName, bool bForceOverwrite = false);
	void FileLoad(std::string &fileName);


	std::string	getFileName();

	// Write to Output File
	void WriteHeader(long lChannels, std::string sChannelMap, long lTemplates, std::string sTemplates, long lInitCt);
	void WriteProcessTime(int processTime);
	void WriteSpike	( long time, long neuron, float amplitude );
	void WriteDecoderInput( int label, std::map<long, double> &window );
	void WritePrediction(int label, int predictLabel, std::vector<double> probabilities, float accuracy, long samplecount);
	void WriteEvent ( t_ull time, int label);


protected:
	// File Name Properties
	std::string		m_sFileName;
	FILE	   	   *m_fFile;
	bool			m_bIsOpen;

	// Helper Functions
	static const std::string currentDateTime();
};

#endif /* FILEWRITER_H_ */
