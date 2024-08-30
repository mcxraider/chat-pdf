export default function Footer() {
  return (
    <footer
      style={{
        padding: "1rem",
        backgroundColor: "#333",
        color: "white",
        marginTop: "auto",
        display: "flex",            // Enable flexbox
        justifyContent: "center",   // Horizontally center the content
        alignItems: "center",       // Vertically center the content
        textAlign: "center",        // Ensure text is centered (useful for multi-line text)
      }}
    >
      <p>&copy; 2024 Chat-pdf</p>
    </footer>
  );
}
