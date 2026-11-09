#ifndef SDM_PARAMS_H_
#define SDM_PARAMS_H_

// Typed read access to a processor's own settings (InputParameters::mapSdmParams).

#include <map>
#include <string>
#include <iostream>

class SdmParams {
public:
	explicit SdmParams(const std::map<std::string, std::string>& values) : m_values(values) {}

	bool has(const std::string& key) const { return m_values.find(key) != m_values.end(); }

	std::string getString(const std::string& key, const std::string& def = "") const {
		auto it = m_values.find(key);
		return it == m_values.end() ? def : it->second;
	}

	double getDouble(const std::string& key, double def) const {
		auto it = m_values.find(key);
		if (it == m_values.end() || it->second.empty())
			return def;
		try { return std::stod(it->second); }
		catch (...) { warn(key, it->second); return def; }
	}

	float getFloat(const std::string& key, float def) const {
		return static_cast<float>(getDouble(key, def));
	}

	int getInt(const std::string& key, int def) const {
		auto it = m_values.find(key);
		if (it == m_values.end() || it->second.empty())
			return def;
		try { return std::stoi(it->second); }
		catch (...) { warn(key, it->second); return def; }
	}

private:
	static void warn(const std::string& key, const std::string& value) {
		std::cerr << "[SDM] Could not parse sdm_param " << key << "='" << value
		          << "'; using the default." << std::endl;
	}

	const std::map<std::string, std::string>& m_values;
};

#endif /* SDM_PARAMS_H_ */
