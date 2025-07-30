/*
 * FileWriter.cpp
 *
 *  Created on: Feb 22, 2019
 *      Author: basti
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <numeric>
#include "Utils.h"

#include "FileWriter.h"

FileWriter::FileWriter()
			: m_sFileName 	("tmp.csv")
			, m_bIsOpen		(   false )
			, m_fFile		(    NULL )
{

}

FileWriter::~FileWriter()
{
	FileClose();
}


bool FileWriter::FileInit( std::string &sFileName, bool bForceOverwrite)
{
	static const char *ptLabel = { "FileWriter::FileInit" };

	if ( m_bIsOpen ) {
		_DEBUG_PUT_0(ptLabel, "Warning: File already open! Closing...");
		FileClose();
	}

	bool bFileExists = false;
	if (FILE *fTmp = fopen(sFileName.c_str(), "r")) {
		fclose( fTmp );
		bFileExists = true;
	}

	if (bFileExists && !bForceOverwrite)
		_DEBUG_PUT_0(ptLabel, "Warning: File " << sFileName.c_str() << " already exists!");

	if ( !bFileExists || bForceOverwrite )
		m_sFileName = sFileName;

	return ( !bFileExists || bForceOverwrite );
}

void FileWriter::FileLoad(std::string &fileName) {
	bool temp = true;
	try
	{
		if (!(FileInit(fileName, false))) {
			std::cout << fileName << " already exists. Do you want to overwrite " << fileName << "?" << std::endl;
			if (temp) {
				if (!FileInit(fileName, true))
					throw std::runtime_error("FileLoad: Failed to load file" + fileName);
			}
			else
				throw std::runtime_error("FileLoad: Failed to load file" + fileName);
		}

		if (!FileOpen())
			throw std::runtime_error("FileLoad: Failed to load file" + fileName);

	}
	catch (std::exception& e) {
		std::cout << "Error while trying to load " << fileName << ": " << e.what() << std::endl;
		exit(1);
	}

	std::cout << "Loaded File: " << getFileName() << std::endl;
}


bool FileWriter::FileOpen()
{
	static const char *ptLabel = { "FileWriter::FileOpen" };

	if ( m_bIsOpen ) {
		_DEBUG_PUT_0(ptLabel, "Warning: File already open! Closing...");
		FileClose();
	}

	m_fFile = fopen( m_sFileName.c_str(), "w" );

	if ( m_fFile )
		m_bIsOpen = true;

	return m_bIsOpen;
}


void FileWriter::FileClose()
{
	static const char *ptLabel = { "FileWriter::FileClose" };

	if ( !m_bIsOpen )
		return;

	fclose( m_fFile );
	m_bIsOpen = false;
}

void FileWriter::WriteHeader(long lChannels, std::string sChannelMap, long lTemplates, std::string sTemplates, long lInitCt) // TODO Use this type of thign for log writing
{
	static const char *ptLabel = { "FileWriter::WriteHeader" };
	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}

	std::string sCTime = currentDateTime();
	fprintf( m_fFile, "########### Spike Output from Online Spike Tool ###########\n" );
	fprintf( m_fFile, "Time:			%s\n", sCTime.c_str() );
	fprintf( m_fFile, "Channel #:		%ld\n", lChannels);
	fprintf( m_fFile, "Channel Map:		%s\n", sChannelMap.c_str() );
	fprintf( m_fFile, "Template #:		%ld\n", lTemplates );
	fprintf( m_fFile, "Template File:	%s\n", sTemplates.c_str() );
	fprintf( m_fFile, "Initial Count:	%ld\n", lInitCt );
	fprintf( m_fFile, "###########################################################\n\n" );

	fprintf( m_fFile, "Time, Template, Amplitude\n" );
}

void FileWriter::WriteProcessTime(int processTime)
{
	static const char *ptLabel = { "FileWriter::WriteProcessTime" };
	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}

	fprintf(m_fFile, "%d\n", processTime);
	fflush(m_fFile);
}


void FileWriter::WriteSpike( long time, long neuron, float amplitude )
{
	static const char *ptLabel = { "FileWriter::WriteSpike" };
	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}
	
	fprintf( m_fFile, "%20ld,%4ld,%f\n", time, neuron, amplitude );
}

void FileWriter::WriteDecoderInput( int label, std::map<long, double> &window ) {
	static const char *ptLabel = { "FileWriter::WriteSpike" };
	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}

	// Write window of spike data in SVMLight format
	fprintf(m_fFile, "+%d", label);
	for (auto const&[channel, count] : window)
		fprintf(m_fFile, " %d:%f", channel, count); // Channel +1 because libSVM is 1-indexed REMEMBER THIS
	fprintf(m_fFile, "\n");
	fflush(m_fFile);
}

void FileWriter::WritePrediction(int label, int predictLabel, std::vector<double> probabilities, float accuracy, long samplecount) {
	static const char *ptLabel = { "FileWriter::WritePrediction" };
	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}

	//fprintf(m_fFile, "%d %d %f %ld  ", label, predictLabel, accuracy, samplecount);
	fprintf(m_fFile, "%ld %d", samplecount, predictLabel);

	if (accumulate(probabilities.begin(), probabilities.end(), 0.0) != 0.0) { // Only output probabilities if predict_probability
		for (auto const& prob : probabilities)
			fprintf(m_fFile, " %f", prob);
	}
	fprintf(m_fFile, "\n");
	fflush(m_fFile);
}

void FileWriter::WriteEvent(t_ull time, int label)
{
	static const char *ptLabel = { "FileWriter::WriteEvent" };

	if (!m_bIsOpen) {
		_DEBUG_PUT_0(ptLabel, "Attempting to write, but file not opened!");
		return;
	}

	fprintf( m_fFile, "%llu %d\n", time, label);
	fflush(m_fFile);
}


std::string FileWriter::getFileName( )
{
	return m_sFileName;
}


const std::string FileWriter::currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return buf;
}
