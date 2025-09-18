import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

// Function to convert plain URLs to markdown link format and fix malformed links
const convertUrlsToMarkdownLinks = (text) => {
  if (!text) return text;
  
  // Fix malformed markdown links that have extra parentheses
  // Pattern: [text](url)) -> [text](url)
  text = text.replace(/\]\(([^)]+)\)\)/g, ']($1)');
  
  // Fix cases where link text might be missing or malformed
  // Pattern: [](url) or [text](url) where text is empty
  text = text.replace(/\[\s*\]\(([^)]+)\)/g, (match, url) => {
    // Extract a meaningful text from the URL
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(part => part);
    const meaningfulText = pathParts[pathParts.length - 1] || urlObj.hostname;
    return `[${meaningfulText}](${url})`;
  });
  
  // Regular expression to match URLs that are not already in markdown format
  const urlRegex = /(?<!\]\()(https?:\/\/[^\s\]]+)(?!\))/g;
  
  // Replace standalone URLs with markdown link format
  return text.replace(urlRegex, (url) => {
    // Clean up the URL (remove trailing punctuation that might not be part of the URL)
    const cleanUrl = url.replace(/[.,;:!?]+$/, '');
    const punctuation = url.slice(cleanUrl.length);
    
    // Create meaningful link text from URL
    try {
      const urlObj = new URL(cleanUrl);
      const pathParts = urlObj.pathname.split('/').filter(part => part);
      const meaningfulText = pathParts[pathParts.length - 1] || urlObj.hostname;
      return `[${meaningfulText}](${cleanUrl})${punctuation}`;
    } catch (e) {
      // If URL parsing fails, use the URL itself as text
      return `[${cleanUrl}](${cleanUrl})${punctuation}`;
    }
  });
};

// Safe ReactMarkdown component with error handling
const SafeMarkdown = ({ children, ...props }) => {
  try {
    // Preprocess text to convert plain URLs to markdown links
    const processedText = convertUrlsToMarkdownLinks(children);
    
    return (
      <ReactMarkdown
        components={{
          h1: ({children}) => <h1 style={{fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#333'}}>{children}</h1>,
          h2: ({children}) => <h2 style={{fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#333'}}>{children}</h2>,
          h3: ({children}) => <h3 style={{fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#333'}}>{children}</h3>,
          strong: ({children}) => <strong style={{fontWeight: 'bold', color: '#2c3e50'}}>{children}</strong>,
          p: ({children}) => <p style={{marginBottom: '0.75rem'}}>{children}</p>,
          ul: ({children}) => <ul style={{marginBottom: '0.75rem', paddingLeft: '1.5rem'}}>{children}</ul>,
          ol: ({children}) => <ol style={{marginBottom: '0.75rem', paddingLeft: '1.5rem'}}>{children}</ol>,
          li: ({children}) => <li style={{marginBottom: '0.25rem'}}>{children}</li>,
          code: ({children}) => <code style={{background: '#e9ecef', padding: '0.2rem 0.4rem', borderRadius: '3px', fontFamily: 'monospace'}}>{children}</code>,
           a: ({href, children}) => {
             // Ensure children is not empty and href is valid
             const linkText = children && children.toString().trim() ? children : href;
             return (
               <a 
                 href={href} 
                 target="_blank" 
                 rel="noopener noreferrer" 
                 style={{
                   color: '#007bff', 
                   textDecoration: 'underline',
                   cursor: 'pointer'
                 }}
                 title={href}
               >
                 {linkText}
               </a>
             );
           }
        }}
        {...props}
      >
        {processedText}
      </ReactMarkdown>
    );
  } catch (error) {
    console.error('Error rendering markdown:', error);
    // Fallback to plain text if markdown fails
    return <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{children}</div>;
  }
};

const ALLOWED_MIME = ["audio/mpeg","audio/wav"];
const ALLOWED_EXT = [".mp3",".wav"];

function formatBytes(b) {
  if (!b) return "0 B";
  const k = 1024,
    s = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(b) / Math.log(k));
  return `${(b / Math.pow(k, i)).toFixed(2)} ${s[i]}`;
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [active, setActive] = useState(null);
  const [errors, setErrors] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadCancelToken, setUploadCancelToken] = useState(null);
  
  // Analysis states
  const [activeTab, setActiveTab] = useState("business");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState("");
  const [backendStatus, setBackendStatus] = useState("unknown");
  const [analysisCancelToken, setAnalysisCancelToken] = useState(null);
  
  // Step-by-step analysis states
  const [currentStep, setCurrentStep] = useState(null); // 'transcription', 'intent', 'keywords', 'resolution'
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [intentResult, setIntentResult] = useState(null);
  const [keywordResult, setKeywordResult] = useState(null);
  const [resolutionResult, setResolutionResult] = useState(null);
  const [stepError, setStepError] = useState(null);
  
  // Backend Collections states
  const [collections, setCollections] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  
  // STT Model states
  const [currentSttModel, setCurrentSttModel] = useState("whisper");
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [showSttDropdown, setShowSttDropdown] = useState(false);

  const inputRef = useRef(null);
  const collectionListRef = useRef(null);
  const dropZoneRef = useRef(null);

  // Check backend health and fetch collection data once on mount
  useEffect(() => {
    try {
      checkBackendHealth();
      fetchCurrentSttModel();
    } catch (error) {
      console.error("Error in initial setup:", error);
      setBackendStatus("disconnected");
    }
    
    const fetchCollections = (forceRefresh = false) => {
      try {
        // Add cache busting parameter
        const timestamp = new Date().getTime();
        const url = forceRefresh 
          ? `http://localhost:8000/collection/all?t=${timestamp}&refresh=true`
          : `http://localhost:8000/collection/all?t=${timestamp}`;
          
        fetch(url, {
          method: 'GET',
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
          }
        })
          .then((res) => {
            if (!res.ok) {
              throw new Error(`HTTP error! status: ${res.status}`);
            }
            return res.json();
          })
          .then((data) => {
            console.log("Collections loaded:", data);
            setCollections(data);
            
            // If we got a timeout, retry after 5 seconds
            if (data.status === "timeout") {
              console.log("Database timeout detected, retrying in 5 seconds...");
              setTimeout(() => fetchCollections(true), 5000);
            }
          })
          .catch((err) => {
            console.error("Error fetching collections:", err);
            setCollections({ data: [], status: "error", message: err.message });
          });
      } catch (error) {
        console.error("Error in fetchCollections:", error);
        setCollections({ data: [], status: "error", message: "Failed to fetch collections" });
      }
    };
    
    // Initial fetch
    fetchCollections();
    
    // Set up automatic refresh every 30 seconds
    const refreshInterval = setInterval(() => {
      console.log("Auto-refreshing collections...");
      fetchCollections(true);
    }, 30000); // 30 seconds
    
    // Add focus event listener to refresh when user returns to tab
    const handleFocus = () => {
      console.log("Window focused, refreshing collections...");
      fetchCollections(true);
    };
    
    window.addEventListener('focus', handleFocus);
    
    // Cleanup interval and event listener on unmount
    return () => {
      clearInterval(refreshInterval);
      window.removeEventListener('focus', handleFocus);
    };
  }, []);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      // Cancel any ongoing analysis when component unmounts
      if (analysisCancelToken) {
        analysisCancelToken.cancel('Component unmounting');
      }
      // Cancel any ongoing upload when component unmounts
      if (uploadCancelToken) {
        uploadCancelToken.cancel('Component unmounting');
      }
    };
  }, [analysisCancelToken, uploadCancelToken]);

  const validateFiles = (fileList) => {
    const next = [],
      errs = [];
    [...fileList].forEach((f) => {
      const okExt = ALLOWED_EXT.some((ext) =>
        f.name.toLowerCase().endsWith(ext)
      );
      const okMime = ALLOWED_MIME.includes(f.type);
      if (!(okExt || okMime))
        errs.push(`Invalid: ${f.name} (only .mp3, .wav allowed)`);
      else next.push(f);
    });
    return { next, errs };
  };

  const onPick = () => inputRef.current?.click();

  const onSelect = (e) => {
    setMessage("");
    setUploadStatus("");
    setAnalysisResult(null);
    setAnalysisError("");
    const { next, errs } = validateFiles(e.target.files);
    setFiles(next);
    setActive(next?.[0]?.name ?? null);
    setErrors(errs);
  };

  // Remove individual file
  const removeFile = (fileName) => {
    const updatedFiles = files.filter(f => f.name !== fileName);
    setFiles(updatedFiles);
    if (active === fileName) {
      setActive(updatedFiles.length > 0 ? updatedFiles[0].name : null);
    }
    setMessage("");
    setUploadStatus("");
  };

  // Clear all files
  const clearAllFiles = () => {
    setFiles([]);
    setActive(null);
    setMessage("");
    setUploadStatus("");
    setErrors([]);
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    setMessage("");
    setUploadStatus("");
    setAnalysisResult(null);
    setAnalysisError("");
    
    const droppedFiles = e.dataTransfer.files;
    const { next, errs } = validateFiles(droppedFiles);
    
    // Add new files to existing ones
    setFiles(prevFiles => [...prevFiles, ...next]);
    setActive(next.length > 0 ? next[0].name : active);
    setErrors(errs);
  };

  // Cancel upload
  const cancelUpload = () => {
    if (uploadCancelToken) {
      uploadCancelToken.cancel('Upload cancelled by user');
      setUploadCancelToken(null);
    }
    setIsUploading(false);
    setProgress(0);
    setUploadStatus("Upload cancelled");
  };

  const onUpload = async () => {
    if (!files.length) return;
    setIsUploading(true);
    setProgress(0);
    setMessage("");
    setUploadStatus("Preparing upload...");
    setErrors([]);

    // Create cancel token
    const cancelToken = axios.CancelToken.source();
    setUploadCancelToken(cancelToken);

    try {
      const fd = new FormData();
      files.forEach((f) => {
        console.log(`Adding file to FormData: ${f.name} (${f.size} bytes)`);
        fd.append("files", f);
      });
      setProgress(5);
      setUploadStatus(`Uploading ${files.length} file(s)...`);
      
      // Add cache busting to upload URL
      const timestamp = new Date().getTime();
      const uploadUrl = `http://localhost:8000/api/upload/audio?t=${timestamp}`;
      
      console.log(`Uploading to: ${uploadUrl}`);
      const uploadResponse = await axios.post(uploadUrl, fd, {
        headers: { 
          "Content-Type": "multipart/form-data",
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache",
          "Expires": "0"
        },
        cancelToken: cancelToken.token,
        onUploadProgress: (evt) => {
          if (!evt.total) return;
          const progressPercent = Math.round((evt.loaded * 100) / evt.total);
          setProgress(progressPercent);
          setUploadStatus(`Uploading... ${progressPercent}%`);
        },
      });
      
      console.log("Upload response:", uploadResponse.data);
      
      setProgress(100);
      setUploadStatus("Upload completed successfully!");
      setMessage("✅ File(s) uploaded successfully.");
      
      // Clear files after successful upload
      setFiles([]);
      setActive(null);
      
      // Refetch collections after upload with a small delay to ensure backend processing is complete
      setTimeout(() => {
        const timestamp = new Date().getTime();
        fetch(`http://localhost:8000/collection/all?t=${timestamp}&refresh=true`, {
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
          }
        })
          .then((res) => res.json())
          .then((data) => {
            console.log("Collections refreshed after upload:", data);
            setCollections(data);
          })
          .catch((err) => console.error("Error fetching collections:", err));
      }, 1000);
        
    } catch (err) {
      if (axios.isCancel(err)) {
        setUploadStatus("Upload cancelled");
        setMessage("⚠️ Upload was cancelled.");
      } else {
        setUploadStatus("Upload failed");
        setMessage("❌ Upload failed.");
        setErrors([err?.response?.data?.message || err.message || "Unknown error"]);
      }
      setProgress(0);
    } finally {
      setUploadCancelToken(null);
      setTimeout(() => {
        setIsUploading(false);
        setUploadStatus("");
      }, 2000);
    }
  };

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get("http://localhost:8000/", {
        timeout: 5000 // 5 second timeout
      });
      console.log("Backend health check:", response.data);
      setBackendStatus("connected");
      return true;
    } catch (error) {
      console.error("Backend not reachable:", error.message);
      setBackendStatus("disconnected");
      return false;
    }
  };

  // Function to fetch audio data for a specific point ID
  const fetchAudioData = async (pointId) => {
    setIsLoadingAudio(true);
    try {
      const timestamp = new Date().getTime();
      const response = await fetch(`http://localhost:8000/collection/audio/${pointId}?t=${timestamp}`, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });
      if (response.ok) {
        const data = await response.json();
        setAudioData(data);
        return data;
      } else {
        console.error('Failed to fetch audio data:', response.statusText);
        return null;
      }
    } catch (error) {
      console.error('Error fetching audio data:', error);
      return null;
    } finally {
      setIsLoadingAudio(false);
    }
  };

  // Function to convert base64 to File object for API calls
  const base64ToFile = (base64String, filename) => {
    try {
      console.log(`Converting base64 to file: ${filename}`);
      console.log(`Base64 string length: ${base64String?.length}`);
      
      // Clean the base64 string - remove any whitespace, newlines, or data URL prefix
      let cleanBase64 = base64String;
      
      // Remove data URL prefix if present (e.g., "data:audio/mp3;base64,")
      if (cleanBase64.includes(',')) {
        cleanBase64 = cleanBase64.split(',')[1];
      }
      
      // Remove any whitespace characters (spaces, newlines, tabs)
      cleanBase64 = cleanBase64.replace(/\s/g, '');
      
      // Ensure the string length is a multiple of 4 by padding with '=' if needed
      while (cleanBase64.length % 4 !== 0) {
        cleanBase64 += '=';
      }
      
      // Convert base64 to binary
      const byteCharacters = atob(cleanBase64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      
      // Determine MIME type based on filename
      let mimeType = 'audio/mpeg'; // default
      if (filename.toLowerCase().endsWith('.wav')) {
        mimeType = 'audio/wav';
      } else if (filename.toLowerCase().endsWith('.mp3')) {
        mimeType = 'audio/mpeg';
      }
      
      const file = new File([byteArray], filename, { type: mimeType });
      console.log(`Created file: ${file.name}, size: ${file.size} bytes, type: ${file.type}`);
      return file;
    } catch (error) {
      console.error('Error converting base64 to file:', error);
      console.error('Base64 string length:', base64String?.length);
      console.error('Base64 string preview:', base64String?.substring(0, 100));
      throw new Error(`Failed to convert base64 audio data: ${error.message}`);
    }
  };


  // Function to run step-by-step analysis
  const runStepByStepAnalysis = async (audioDataItem) => {
    console.log('runStepByStepAnalysis called for:', audioDataItem.source);
    
    // Cancel any ongoing analysis first
    if (analysisCancelToken) {
      console.log('Cancelling existing analysis before starting new one');
      analysisCancelToken.cancel('Starting new analysis');
      setAnalysisCancelToken(null);
    }
    
    // Create new cancel token immediately
    const cancelToken = axios.CancelToken.source();
    setAnalysisCancelToken(cancelToken);

    setIsAnalyzing(true);
    setAnalysisError("");
    setAnalysisResult(null);
    setStepError("");
    
    // Reset step results
    setTranscriptionResult(null);
    setIntentResult(null);
    setKeywordResult(null);
    setResolutionResult(null);
    setCurrentStep(null);


    try {
      // Validate audio item
      if (!audioDataItem?.audio_base64) {
        throw new Error("No audio data found in the selected item");
      }

      console.log('Audio data item:', {
        source: audioDataItem.source,
        original_filename: audioDataItem.original_filename,
        audio_base64_length: audioDataItem.audio_base64?.length
      });

      // Convert base64 audio back to File
      const audioFile = base64ToFile(audioDataItem.audio_base64, audioDataItem.source);
      
      console.log('Created audio file:', {
        name: audioFile.name,
        size: audioFile.size,
        type: audioFile.type
      });
      
      // Step 1: Transcription
      setCurrentStep('transcription');
      console.log('Starting transcription...');
      
      // Check if cancelled before starting
      if (cancelToken.token.reason) {
        console.log('Analysis cancelled before transcription');
        return;
      }
      
      const transcriptionFormData = new FormData();
      transcriptionFormData.append('file', audioFile);
      
      const timestamp = new Date().getTime();
      const transcriptionResponse = await axios.post(`http://localhost:8000/transcribe?t=${timestamp}`, transcriptionFormData, {
        headers: { 
          'Content-Type': 'multipart/form-data',
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        },
        timeout: 300000, // 5 minute timeout for large files
        cancelToken: cancelToken.token,
      });

      console.log('Transcription response:', transcriptionResponse.data);

      if (transcriptionResponse.data.success) {
        setTranscriptionResult(transcriptionResponse.data);
        console.log('Transcription completed successfully');
        
        // Step 2: Intent Recognition
        setCurrentStep('intent');
        console.log('Starting intent recognition...');
        
        // Check if cancelled before starting
        if (cancelToken.token.reason) {
          console.log('Analysis cancelled before intent recognition');
          return;
        }
        
        const intentTimestamp = new Date().getTime();
        const intentResponse = await axios.post(`http://localhost:8000/recognize-intent?t=${intentTimestamp}`, {
          text: transcriptionResponse.data.transcription
        }, {
          headers: { 
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
          },
          timeout: 120000, // 2 minute timeout
          cancelToken: cancelToken.token,
        });

        console.log('Intent response:', intentResponse.data);

        if (intentResponse.data.success) {
          setIntentResult(intentResponse.data);
          console.log('Intent recognition completed successfully');
          
          // Step 3: Keyword & Entity Extraction
          setCurrentStep('keywords');
          console.log('Starting keyword extraction...');
          
          // Check if cancelled before starting
          if (cancelToken.token.reason) {
            console.log('Analysis cancelled before keyword extraction');
            return;
          }
          
          const keywordTimestamp = new Date().getTime();
          const keywordResponse = await axios.post(`http://localhost:8000/extract-keywords?t=${keywordTimestamp}`, {
            text: transcriptionResponse.data.transcription
          }, {
            headers: { 
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache, no-store, must-revalidate',
              'Pragma': 'no-cache',
              'Expires': '0'
            },
            timeout: 120000, // 2 minute timeout
            cancelToken: cancelToken.token,
          });

          console.log('Keyword extraction response:', keywordResponse.data);

          if (keywordResponse.data.success) {
            setKeywordResult(keywordResponse.data);
            console.log('Keyword extraction completed successfully');
          } else {
            console.log('Keyword extraction failed, continuing with resolution...');
                setKeywordResult({
                    keywords: [],
                    overall_confidence: 0,
                    success: false,
                    message: "Keyword extraction failed"
                });
          }
          
          // Step 4: Resolution Generation
          setCurrentStep('resolution');
          console.log('Starting resolution generation...');
          
          // Check if cancelled before starting
          if (cancelToken.token.reason) {
            console.log('Analysis cancelled before resolution generation');
            return;
          }
          
          const resolutionTimestamp = new Date().getTime();
          const resolutionResponse = await axios.post(`http://localhost:8000/generate-resolution?t=${resolutionTimestamp}`, {
            content: transcriptionResponse.data.transcription,
            intent: intentResponse.data.intent
          }, {
            headers: { 
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache, no-store, must-revalidate',
              'Pragma': 'no-cache',
              'Expires': '0'
            },
            timeout: 300000, // 5 minute timeout
            cancelToken: cancelToken.token,
          });

          console.log('Resolution response:', resolutionResponse.data);

          if (resolutionResponse.data.success) {
            setResolutionResult(resolutionResponse.data);
            console.log('Resolution generation completed successfully');
            
            // Create final analysis result
            const finalResult = {
              transcription: transcriptionResponse.data.transcription,
              intent: intentResponse.data.intent,
                keywords: keywordResult || { keywords: [], overall_confidence: 0 },
              resolution: resolutionResponse.data.resolution,
              success: true,
              message: "Step-by-step analysis with keyword extraction completed successfully"
            };
            setAnalysisResult(finalResult);
            setCurrentStep(null);
            setIsAnalyzing(false);
            console.log('All steps completed successfully for:', audioDataItem.source);
            console.log('Setting isAnalyzing to false, currentStep to null');
          } else {
            setStepError("Resolution generation failed: " + (resolutionResponse.data.message || "Unknown error"));
            setIsAnalyzing(false);
          }
        } else {
          setStepError("Intent recognition failed: " + (intentResponse.data.message || "Unknown error"));
          setIsAnalyzing(false);
        }
      } else {
        setStepError("Transcription failed: " + (transcriptionResponse.data.message || "Unknown error"));
        setIsAnalyzing(false);
      }
    } catch (error) {
      console.error('Step-by-step analysis error:', error);
      
      // Handle cancellation
      if (axios.isCancel(error)) {
        console.log('Analysis was cancelled for:', audioDataItem.source);
        setAnalysisError("Analysis cancelled");
        return; // Don't set isAnalyzing to false here, let the finally block handle it
      }
      
      // Handle different types of errors
      let errorMessage = "Analysis failed";
      
      if (error.message && error.message.includes("Failed to convert base64")) {
        errorMessage = "Invalid audio data format. Please try re-uploading the audio file.";
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setStepError(errorMessage);
    } finally {
      // Only reset if this is still the current analysis (not cancelled)
      if (analysisCancelToken === cancelToken) {
        setIsAnalyzing(false);
        setAnalysisCancelToken(null);
        setCurrentStep(null);
      }
    }
  };

  // Handle audio selection and auto-analysis
  const handleAudioSelection = async (audioItem) => {
    // Cancel ongoing analysis immediately and aggressively
    if (analysisCancelToken) {
      console.log('Immediately cancelling previous analysis for:', selectedAudio?.payload?.source);
      analysisCancelToken.cancel('User switched to different audio file');
      setAnalysisCancelToken(null);
    }
  
    // Reset all states immediately to show new file is selected
    setSelectedAudio(audioItem);
    setAudioData(null);
    setAnalysisResult(null);
    setTranscriptionResult(null);
    setIntentResult(null);
    setKeywordResult(null);
    setResolutionResult(null);
    setAnalysisError("");
    setStepError("");
    setIsAnalyzing(false);
    setCurrentStep(null);
  
    console.log('Starting analysis for new file:', audioItem.payload?.source);
    
    // Fetch new audio data and start analysis immediately
    const fetchedAudioData = await fetchAudioData(audioItem.id);
    if (fetchedAudioData) {
      // Start new analysis immediately without waiting
      runStepByStepAnalysis(fetchedAudioData);
    }
  };

  // Scroll to top button handler
  const scrollToTop = () => {
    if (collectionListRef.current) {
      collectionListRef.current.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  // STT Model switching functions
  const fetchCurrentSttModel = async () => {
    try {
      const response = await axios.get('http://localhost:8000/stt-model');
      if (response.data.success) {
        setCurrentSttModel(response.data.model);
      }
    } catch (error) {
      console.error('Error fetching current STT model:', error);
    }
  };

  const switchSttModel = async (newModel) => {
    if (isSwitchingModel) return;
    
    setIsSwitchingModel(true);
    try {
      const response = await axios.post('http://localhost:8000/stt-model', {
        model: newModel
      });
      
      if (response.data.success) {
        setCurrentSttModel(response.data.model);
        console.log(`STT model switched to: ${response.data.model}`);
      } else {
        console.error('Failed to switch STT model:', response.data.message);
      }
    } catch (error) {
      console.error('Error switching STT model:', error);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  // Show loading state if backend status is unknown
  if (backendStatus === "unknown") {
    return (
      <div className="app">
        <div style={{ 
          display: "flex", 
          justifyContent: "center", 
          alignItems: "center", 
          height: "100vh",
          flexDirection: "column",
          gap: "20px"
        }}>
          <div className="spinner" style={{ width: "40px", height: "40px" }}></div>
          <div style={{ color: "#666", fontSize: "16px" }}>Loading AI Voice Analytics...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Hidden STT Model Selection - Clickable area in top-right corner */}
      <div style={{ 
        position: "fixed",
        top: "10px",
        right: "10px",
        width: "20px",
        height: "20px",
        cursor: "pointer",
        zIndex: 1000,
        background: "transparent"
      }}
      onClick={() => setShowSttDropdown(!showSttDropdown)}
      title="STT Model Selection"
      >
        {/* Invisible clickable area */}
      </div>

      {/* Dropdown that appears when clicked */}
      {showSttDropdown && (
        <div style={{
          position: "fixed",
          top: "35px",
          right: "10px",
          background: "#ffffff",
          border: "1px solid #ddd",
          borderRadius: "6px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
          zIndex: 1001,
          minWidth: "150px",
          padding: "8px 0"
        }}>
          <div style={{
            padding: "8px 12px",
            fontSize: "12px",
            fontWeight: "600",
            color: "#666",
            borderBottom: "1px solid #eee",
            marginBottom: "4px"
          }}>
            STT Model
          </div>
          <div
            style={{
              padding: "8px 12px",
              cursor: "pointer",
              fontSize: "14px",
              color: currentSttModel === "kibo" ? "#007bff" : "#666",
              background: currentSttModel === "kibo" ? "#f8f9fa" : "transparent",
              fontWeight: currentSttModel === "kibo" ? "600" : "400"
            }}
            onClick={() => switchSttModel("kibo")}
          >
            {isSwitchingModel && currentSttModel === "kibo" ? "Switching..." : "Kibo AI"}
          </div>
          <div
            style={{
              padding: "8px 12px",
              cursor: "pointer",
              fontSize: "14px",
              color: currentSttModel === "whisper" ? "#28a745" : "#666",
              background: currentSttModel === "whisper" ? "#f8f9fa" : "transparent",
              fontWeight: currentSttModel === "whisper" ? "600" : "400"
            }}
            onClick={() => switchSttModel("whisper")}
          >
            {isSwitchingModel && currentSttModel === "whisper" ? "Switching..." : "Whisper"}
          </div>
          <div
            style={{
              padding: "8px 12px",
              cursor: "pointer",
              fontSize: "14px",
              color: currentSttModel === "assisto" ? "#ff6b35" : "#666",
              background: currentSttModel === "assisto" ? "#f8f9fa" : "transparent",
              fontWeight: currentSttModel === "assisto" ? "600" : "400"
            }}
            onClick={() => switchSttModel("assisto")}
          >
            {isSwitchingModel && currentSttModel === "assisto" ? "Switching..." : "Assisto"}
          </div>
        </div>
      )}

      {/* Click outside to close dropdown */}
      {showSttDropdown && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 999,
            background: "transparent"
          }}
          onClick={() => setShowSttDropdown(false)}
        />
      )}

      {/* Sidebar */}
      <aside className="sidebar" style={{ height: "100vh", overflowY: "auto", display: "flex", flexDirection: "column" }}>
        <h1>AI Voice Analytics PoC</h1>
        

        <input
          ref={inputRef}
          id="audio-input"
          type="file"
          accept=".mp3,audio/mpeg,.wav"
          multiple
          className="hidden-input"
          onChange={onSelect}
        />

        {/* Drag and Drop Zone */}
        <div
          ref={dropZoneRef}
          className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${files.length > 0 ? 'has-files' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={onPick}
        >
          <div className="drop-zone-content">
            <div className="drop-zone-icon">🎵</div>
            <div className="drop-zone-text">
              {isDragOver ? 'Drop audio files here' : 'Click to select or drag & drop audio files'}
            </div>
            <div className="drop-zone-helper">Allowed: .mp3, .wav files (multiple files supported)</div>
          </div>
        </div>

        {/* File List with improved UI - Compact version */}
        {files.length > 0 && (
          <div style={{ marginBottom: "10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={{ fontSize: "14px", fontWeight: "500" }}>Selected Files ({files.length})</span>
              <button 
                onClick={clearAllFiles}
                style={{
                  background: "#dc3545",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  padding: "2px 6px",
                  fontSize: "11px",
                  cursor: "pointer"
                }}
                title="Clear all files"
              >
                ✕ Clear All
              </button>
            </div>
            <div style={{ maxHeight: "120px", overflowY: "auto", background: "#f8f9fa", borderRadius: "4px", padding: "8px" }}>
              {files.map((f) => (
                <div
                  key={f.name}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    padding: "4px 0",
                    borderBottom: "1px solid #e9ecef",
                    fontSize: "12px"
                  }}
                >
                  <div style={{ marginRight: "8px" }}>🎵</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ 
                      fontWeight: active === f.name ? "600" : "400",
                      color: active === f.name ? "#007bff" : "#333",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis"
                    }} title={f.name}>
                      {f.name}
                    </div>
                    <div style={{ fontSize: "10px", color: "#666" }}>{formatBytes(f.size)}</div>
                  </div>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(f.name);
                    }}
                    style={{
                      background: "none",
                      border: "none",
                      color: "#dc3545",
                      cursor: "pointer",
                      padding: "2px",
                      fontSize: "12px"
                    }}
                    title="Remove file"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload Controls - Compact version */}
        {files.length > 0 && (
          <div style={{ marginBottom: "15px" }}>
            <button
              onClick={onUpload}
              disabled={isUploading}
              style={{
                width: "100%",
                padding: "8px 12px",
                background: isUploading ? "#6c757d" : "#007bff",
                color: "white",
                border: "none",
                borderRadius: "4px",
                fontSize: "14px",
                cursor: isUploading ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px"
              }}
            >
              {isUploading && <div style={{ width: "12px", height: "12px", border: "2px solid #fff", borderTop: "2px solid transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }}></div>}
              {isUploading ? "Uploading…" : `Upload ${files.length} file(s)`}
            </button>
            {isUploading && (
              <button
                onClick={cancelUpload}
                style={{
                  width: "100%",
                  marginTop: "8px",
                  padding: "6px 12px",
                  background: "#dc3545",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  fontSize: "12px",
                  cursor: "pointer"
                }}
                title="Cancel upload"
              >
                Cancel Upload
              </button>
            )}
          </div>
        )}

        {/* Upload Progress - Compact version */}
        {isUploading && (
          <div style={{ marginBottom: "15px", padding: "10px", background: "#f8f9fa", borderRadius: "4px", border: "1px solid #e9ecef" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px", fontSize: "12px" }}>
              <span style={{ color: "#666" }}>{uploadStatus}</span>
              <span style={{ fontWeight: "600", color: "#007bff" }}>{progress}%</span>
            </div>
            <div style={{ width: "100%", height: "6px", background: "#e9ecef", borderRadius: "3px", overflow: "hidden" }}>
              <div 
                style={{ 
                  width: `${progress}%`, 
                  height: "100%", 
                  background: "#007bff", 
                  transition: "width 0.3s ease" 
                }}
              />
            </div>
          </div>
        )}

        {/* Status Messages */}
        {message && (
          <div className={`status ${message.includes("failed") || message.includes("cancelled") ? "fail" : "ok"}`}>
            {message}
          </div>
        )}
        
        {uploadStatus && !isUploading && (
          <div className={`status ${uploadStatus.includes("failed") || uploadStatus.includes("cancelled") ? "fail" : "ok"}`}>
            {uploadStatus}
          </div>
        )}

        {/* Error list */}
        {errors.length > 0 && (
          <div className="status fail" role="alert">
            {errors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}

        {/* Backend Collections List */}
        <div style={{ marginTop: 20, flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <h3 style={{ margin: "0 0 10px 0", flexShrink: 0 }}>Backend Collections:</h3>
          <div
            ref={collectionListRef}
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "10px",
              background: "#f9f9f9",
              borderRadius: "6px",
              border: "1px solid #e0e0e0",
              boxSizing: "border-box",
              minHeight: 0
            }}
          >
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {collections && (collections.status === "error" || collections.status === "timeout" || collections.status === "unavailable") ? (
                <li style={{ 
                  padding: "15px", 
                  background: collections.status === "timeout" ? "#fff3cd" : "#ffebee", 
                  color: collections.status === "timeout" ? "#856404" : "#c62828", 
                  borderRadius: "4px",
                  border: `1px solid ${collections.status === "timeout" ? "#ffc107" : "#f44336"}`
                }}>
                  <strong>
                    {collections.status === "timeout" && "Database Timeout:"}
                    {collections.status === "unavailable" && "Service Unavailable:"}
                    {collections.status === "error" && "Error loading collections:"}
                  </strong> {collections.message}
                  <br />
                  <small>
                    {collections.status === "timeout" && "The database is experiencing high load. Please try again in a few moments."}
                    {collections.status === "unavailable" && "The database service is temporarily unavailable. Please try again later."}
                    {collections.status === "error" && "Make sure the backend is running on http://localhost:8000"}
                  </small>
                </li>
              ) : collections && collections.data && collections.data.length > 0 ? (
                collections.data.map((item) => (
                                      <li
                      key={item.id}
                      style={{
                        marginBottom: "12px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        cursor: "pointer",
                        background:
                          selectedAudio && selectedAudio.id === item.id
                            ? isAnalyzing
                              ? "#fff3cd"
                              : "#e8f0fe"
                            : "transparent",
                        borderRadius: "6px",
                        padding: "8px",
                        transition: "background .17s",
                        border: selectedAudio && selectedAudio.id === item.id && isAnalyzing
                          ? "2px solid #ffc107"
                          : "2px solid transparent",
                        boxSizing: "border-box"
                      }}
                      onClick={() => handleAudioSelection(item)}
                    >
                    <span
                      style={{
                        flex: 1,
                        fontWeight: 500,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        minWidth: 0
                      }}
                    >
                      {item.payload?.source || item.id}
                      {selectedAudio && selectedAudio.id === item.id && isAnalyzing && (
                        <span style={{ 
                          fontSize: "10px", 
                          color: "#ffc107", 
                          fontWeight: "bold",
                          background: "#fff3cd",
                          padding: "2px 6px",
                          borderRadius: "10px",
                          border: "1px solid #ffc107"
                        }}>
                          {currentStep === 'transcription' && 'TRANSCRIBING...'}
                          {currentStep === 'intent' && 'ANALYZING INTENT...'}
                          {currentStep === 'keywords' && 'EXTRACTING KEYWORDS...'}
                          {currentStep === 'resolution' && 'GENERATING RESOLUTION...'}
                          {!currentStep && 'ANALYZING...'}
                        </span>
                      )}
                    </span>
                    <div style={{ flex: "0 0 auto", display: "flex", alignItems: "center", minWidth: "120px" }}>
                      {selectedAudio && selectedAudio.id === item.id && isLoadingAudio ? (
                        <div style={{ padding: "8px", color: "#666", fontSize: "11px" }}>Loading...</div>
                      ) : selectedAudio && selectedAudio.id === item.id && audioData ? (
                        <audio
                          src={`data:${audioData.original_filename?.toLowerCase().endsWith('.wav') ? 'audio/wav' : 'audio/mpeg'};base64,${audioData.audio_base64}`}
                          controls
                          style={{ width: "100%", maxWidth: "200px" }}
                        />
                      ) : (
                        <div style={{ padding: "8px", color: "#999", fontSize: "11px" }}>
                          Click to load
                        </div>
                      )}
                    </div>
                  </li>
                ))
              ) : (
                <li>No files found</li>
              )}
            </ul>
          </div>


        </div>
      </aside>

              {/* Content panel */}
        <section className="panel">
          <div className="tabs">
            <div 
              className={`tab ${activeTab === "business" ? "active" : ""}`}
              onClick={() => setActiveTab("business")}
            >
              Business View
            </div>
            <div 
              className={`tab ${activeTab === "diagnostic" ? "active" : ""}`}
              onClick={() => setActiveTab("diagnostic")}
            >
              Diagnostic View
            </div>
          </div>

          <div className="panel-body">
            {selectedAudio ? (
              <div style={{ marginBottom: "22px", padding: "12px 0" }}>
                {/* Audio file name and player - Show in both Business and Diagnostic Views */}
                <div
                  style={{
                    fontWeight: 700,
                    marginBottom: "5px",
                    fontSize: "16px",
                  }}
                >
                  {selectedAudio.payload?.source}
                </div>
                
                {isLoadingAudio ? (
                  <div style={{ 
                    padding: "20px", 
                    textAlign: "center", 
                    background: "#f8f9fa", 
                    borderRadius: "8px",
                    border: "1px solid #e9ecef"
                  }}>
                    Loading audio data...
                  </div>
                ) : audioData ? (
                  <audio
                    src={`data:${audioData.original_filename?.toLowerCase().endsWith('.wav') ? 'audio/wav' : 'audio/mpeg'};base64,${audioData.audio_base64}`}
                    controls
                    style={{ width: "100%" }}
                  />
                ) : (
                  <div style={{ 
                    padding: "20px", 
                    textAlign: "center", 
                    background: "#fff3cd", 
                    borderRadius: "8px",
                    border: "1px solid #ffeaa7",
                    color: "#856404"
                  }}>
                    Audio data will load when you select a file
                  </div>
                )}
                
                
                {/* Analysis Error */}
                {analysisError && (
                  <div style={{ 
                    marginTop: "15px", 
                    padding: "15px", 
                    background: "#ffebee", 
                    borderRadius: "8px",
                    border: "1px solid #f44336",
                    color: "#c62828"
                  }}>
                    <strong>Analysis Error:</strong> {analysisError}
                  </div>
                )}
                
                {/* Step Error */}
                {stepError && (
                  <div style={{ 
                    marginTop: "15px", 
                    padding: "15px", 
                    background: "#ffebee", 
                    borderRadius: "8px",
                    border: "1px solid #f44336",
                    color: "#c62828"
                  }}>
                    <strong>Step Error:</strong> {stepError}
                  </div>
                )}
              </div>
            ) : (
              <div className="kicker">NO FILE</div>
            )}

            {/* Business View */}
            {activeTab === "business" && (
              <>
                {/* Analysis Status - Only show in Business View */}
                {isAnalyzing && currentStep && !(transcriptionResult && intentResult && resolutionResult) && (
                  <div style={{ 
                    marginBottom: "1.5rem", 
                    padding: "15px", 
                    background: "#f0f8ff", 
                    borderRadius: "8px",
                    border: "1px solid #007bff"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                      <div className="spinner" style={{ width: "20px", height: "20px" }}></div>
                      <span style={{ fontWeight: 500 }}>
                        {currentStep === 'transcription' && 'Transcribing audio...'}
                        {currentStep === 'intent' && 'Recognizing intent...'}
                        {currentStep === 'keywords' && 'Extracting keywords...'}
                        {currentStep === 'resolution' && 'Generating resolution...'}
                        {!currentStep && 'Analyzing audio...'}
                      </span>
                      <span style={{ 
                        fontSize: "12px", 
                        color: "#666", 
                        fontStyle: "italic",
                        marginLeft: "auto"
                      }}>
                        Click another file to cancel and switch
                      </span>
                    </div>
                    
                    {/* Step Progress */}
                    <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                      <div style={{ 
                        padding: "4px 8px", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        background: transcriptionResult ? "#d4edda" : currentStep === 'transcription' ? "#fff3cd" : "#e9ecef",
                        color: transcriptionResult ? "#155724" : currentStep === 'transcription' ? "#856404" : "#6c757d",
                        fontWeight: "500"
                      }}>
                        ✓ Transcription
                      </div>
                      <div style={{ 
                        padding: "4px 8px", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        background: intentResult ? "#d4edda" : currentStep === 'intent' ? "#fff3cd" : "#e9ecef",
                        color: intentResult ? "#155724" : currentStep === 'intent' ? "#856404" : "#6c757d",
                        fontWeight: "500"
                      }}>
                        {intentResult ? "✓" : currentStep === 'intent' ? "⏳" : "⏸"} Intent
                      </div>
                      <div style={{ 
                        padding: "4px 8px", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        background: keywordResult ? "#d4edda" : currentStep === 'keywords' ? "#fff3cd" : "#e9ecef",
                        color: keywordResult ? "#155724" : currentStep === 'keywords' ? "#856404" : "#6c757d",
                        fontWeight: "500"
                      }}>
                        {keywordResult ? "✓" : currentStep === 'keywords' ? "⏳" : "⏸"} Keywords
                      </div>
                      <div style={{ 
                        padding: "4px 8px", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        background: resolutionResult ? "#d4edda" : currentStep === 'resolution' ? "#fff3cd" : "#e9ecef",
                        color: resolutionResult ? "#155724" : currentStep === 'resolution' ? "#856404" : "#6c757d",
                        fontWeight: "500"
                      }}>
                        {resolutionResult ? "✓" : currentStep === 'resolution' ? "⏳" : "⏸"} Resolution
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Show completion message when all steps are done */}
                {!isAnalyzing && transcriptionResult && intentResult && keywordResult && resolutionResult && (
                  <div style={{ 
                    marginBottom: "1.5rem", 
                    padding: "15px", 
                    background: "#d4edda", 
                    borderRadius: "8px",
                    border: "1px solid #c3e6cb",
                    color: "#155724"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                      <span style={{ fontSize: "18px" }}>✅</span>
                      <span style={{ fontWeight: 500 }}>Analysis completed successfully!</span>
                    </div>
                  </div>
                )}
                
                {/* Show intent result as soon as it's available */}
                {intentResult && (
                  <div style={{ marginBottom: "1.5rem" }}>
                    <h2 className="title" style={{ color: '#2c3e50' }}>Identified Intent</h2>
                    <h3 className="subheader" style={{ color: '#2c3e50' }}>{(intentResult.intent || "No intent identified").replace(/[{}]/g, '')}</h3>
                  </div>
                )}
                
                {/* Show resolution result as soon as it's available */}
                {resolutionResult && (
                  <div style={{ marginBottom: "1.5rem" }}>
                    <h2 className="title" style={{ color: '#2c3e50' }}>Recommended Resolution Path</h2>
                    <div style={{ 
                      background: "#f8f9fa", 
                      padding: "1rem", 
                      borderRadius: "8px", 
                      marginBottom: "1rem",
                      lineHeight: "1.6"
                    }}>
                      <SafeMarkdown>
                        {resolutionResult.resolution || "No resolution available"}
                      </SafeMarkdown>
                    </div>
                  </div>
                )}
                
                {/* Show final analysis result if available (for backward compatibility) */}
                {analysisResult && !intentResult && !resolutionResult && (
                  <>
                    <h2 className="title" style={{ color: '#2c3e50' }}>Identified Intent</h2>
                    <h3 className="subheader" style={{ color: '#2c3e50' }}>{(analysisResult.intent || "No intent identified").replace(/[{}]/g, '')}</h3>
                    
                    <h2 className="title" style={{ color: '#2c3e50' }}>Recommended Resolution Path</h2>
                    <div style={{ 
                      background: "#f8f9fa", 
                      padding: "1rem", 
                      borderRadius: "8px", 
                      marginBottom: "1rem",
                      lineHeight: "1.6"
                    }}>
                      <SafeMarkdown>
                        {analysisResult.resolution || "No resolution available"}
                      </SafeMarkdown>
                    </div>
                  </>
                )}
                
                {!analysisResult && !intentResult && !resolutionResult && !isAnalyzing && (
                  <div style={{ 
                    textAlign: "center", 
                    padding: "2rem", 
                    color: "#666",
                    fontStyle: "italic"
                  }}>
                    Select an audio file from the collection to analyze it
                  </div>
                )}
              </>
            )}

            {/* Diagnostic View */}
            {activeTab === "diagnostic" && (
              <>
                {(transcriptionResult || intentResult || analysisResult) && (
                  <>
                    <h2 className="title">Diagnostic Information</h2>
                    
                    {/* Show transcription result with confidence */}
                    {(transcriptionResult || analysisResult) && (
                      <>
                        <div style={{ 
                          display: "flex", 
                          justifyContent: "space-between", 
                          alignItems: "center",
                          marginBottom: "8px"
                        }}>
                          <div className="subheader">Full Audio Transcript</div>
                          {(keywordResult?.overall_confidence || analysisResult?.keywords?.overall_confidence) && (
                            <span style={{ 
                              fontSize: "16px", 
                              fontWeight: "600",
                              color: "#007bff",
                              background: "#e7f3ff",
                              padding: "4px 12px",
                              borderRadius: "20px",
                              border: "1px solid #007bff"
                            }}>
                              Confidence: {(keywordResult?.overall_confidence || analysisResult?.keywords?.overall_confidence || 0)}%
                            </span>
                          )}
                        </div>
                        <div style={{ 
                          background: "#f8f9fa", 
                          padding: "1rem", 
                          borderRadius: "8px", 
                          marginBottom: "1rem",
                          fontFamily: "monospace",
                          fontSize: "14px",
                          lineHeight: "1.6",
                          whiteSpace: "pre-wrap"
                        }}>
                          {(transcriptionResult?.transcription || analysisResult?.transcription) || "No transcription available"}
                        </div>
                      </>
                    )}

                    {/* Intent removed from Diagnostic View - only show in Business View */}

                    {/* Show AI Keyword & Entity Extraction */}
                    {(keywordResult || analysisResult?.keywords) && (
                      <>
                        <div className="subheader">AI Keywords & Entity Extraction</div>
                        <div style={{ 
                          background: "#f8f9fa", 
                          padding: "1rem", 
                          borderRadius: "8px", 
                          marginBottom: "1rem",
                          border: "1px solid #e9ecef"
                        }}>
                          {/* Keywords Section */}
                          <div>
                            <div style={{ 
                              display: "flex", 
                              flexWrap: "wrap", 
                              gap: "8px", 
                              alignItems: "center"
                            }}>
                              {((keywordResult?.keywords || analysisResult?.keywords?.keywords || []).map((keyword, index) => (
                                <span
                                  key={index}
                                  style={{
                                    background: keyword.category === 'gst' ? '#e3f2fd' : 
                                               keyword.category === 'tally_feature' ? '#f3e5f5' :
                                               keyword.category === 'business_entity' ? '#e8f5e8' :
                                               keyword.category === 'action' ? '#fff3e0' :
                                               keyword.category === 'issue' ? '#ffebee' : '#f5f5f5',
                                    color: keyword.category === 'gst' ? '#1976d2' : 
                                           keyword.category === 'tally_feature' ? '#7b1fa2' :
                                           keyword.category === 'business_entity' ? '#388e3c' :
                                           keyword.category === 'action' ? '#f57c00' :
                                           keyword.category === 'issue' ? '#d32f2f' : '#555',
                                    padding: "6px 12px",
                                    borderRadius: "16px",
                                    fontSize: "14px",
                                    fontWeight: "500",
                                    border: `1px solid ${keyword.category === 'gst' ? '#bbdefb' : 
                                                         keyword.category === 'tally_feature' ? '#e1bee7' :
                                                         keyword.category === 'business_entity' ? '#c8e6c9' :
                                                         keyword.category === 'action' ? '#ffe0b2' :
                                                         keyword.category === 'issue' ? '#ffcdd2' : '#e0e0e0'}`,
                                    display: "inline-block"
                                  }}
                                >
                                  {keyword.term}
                                </span>
                              )))}
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </>
                )}
                
                {!transcriptionResult && !intentResult && !analysisResult && (
                  <div style={{ 
                    textAlign: "center", 
                    padding: "2rem", 
                    color: "#666",
                    fontStyle: "italic"
                  }}>
                    No diagnostic data available. Run analysis first.
                  </div>
                )}
              </>
            )}

            
          </div>
        </section>
    </div>
  );
}